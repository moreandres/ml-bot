#!/usr/bin/env python3

"""Script that answers predictions of emails with dataset attachments."""

import base64
import io
import logging
import os
import tempfile

from datetime import datetime

from email.message import EmailMessage
from email.mime.text import MIMEText
import email
from email import policy
import typing
import argparse

from textblob import TextBlob  # type: ignore
from dateutil import parser

import pandas
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
from googleapiclient.discovery import build

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder  # type: ignore
from sklearn import model_selection  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.metrics import classification_report  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.model_selection import GridSearchCV  # type: ignore

from sklearn.ensemble import RandomForestClassifier  # type: ignore


log = logging.getLogger(__name__)


def get_gmail_credentials(
    secrets="credentials.json",
    token="token.json",
    auth_scopes=None,
) -> Credentials:
    """
    Get Gmail credentials.
    Get new token using secrets with manual user authorization in a browser, or refresh it.
    Use working directory to read credentials and store token JSON files.
    """

    if auth_scopes is None:
        auth_scopes = ["https://www.googleapis.com/auth/gmail.modify"]

    credentials = None
    updated = False

    log.debug("getting credentials")

    if os.path.exists(token):
        log.debug("found token")
        credentials = Credentials.from_authorized_user_file(token, auth_scopes)

        if credentials.valid:
            log.debug("valid token")
            return credentials

        if credentials.refresh_token:
            log.debug("refreshing token")
            credentials.refresh(Request())
            updated = True
        else:
            log.debug("expired token")
            credentials = None

    if not credentials:
        log.debug("getting new token")
        flow = InstalledAppFlow.from_client_secrets_file(secrets, auth_scopes)
        credentials = flow.run_local_server(port=0)
        updated = True

    if updated:
        log.debug("caching token")
        with open(token, "w", encoding="utf8") as file:
            file.write(credentials.to_json())
        log.debug("cached token")

    log.debug("got credentials")

    return credentials


def split_data(data):
    """Split data and target column. Use last one unless it is well-known."""

    labels = [
        "Survived",
        "class",
        "Class",
        "HiringDecision",
        "PurchaseStatus",
        "RainTomorrow",
        "Potability",
        "GradeClass",
        "label",
    ]

    name = data.columns[-1]

    log.debug("columns are %s", data.columns.to_list())
    log.debug("last column is %s", name)

    for label in labels:
        if label in data.columns:
            log.debug("known target is %s", label)
            name = label

    target = pandas.DataFrame(data[name])
    data.drop(columns=name, inplace=True)

    fill_missing_values(target)

    return data, target


def process_emails(args):
    """Process emails."""

    service = build(
        "gmail",
        "v1",
        credentials=get_gmail_credentials(secrets=args.credentials, token=args.token),
    )

    log.debug("searching messages")
    results = service.users().messages().list(userId="me", q=args.query).execute()

    messages = results.get("messages", [])
    log.debug("processing %d messages", len(messages))

    for message in messages:
        process_email(message, service, args)


def process_email(message, service, args):
    """Process email."""
    message_id = message["id"]
    log.debug("processing message id %s", message_id)

    txt = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )

    payload = txt["payload"]
    headers = {item["name"]: item["value"] for item in payload["headers"]}
    log.debug("headers: %s", headers)
    attachments = [part for part in payload["parts"] if part.get("filename")]
    log.debug("attachments: %s", attachments)

    log.debug("processing %s attachments", len(attachments))

    with tempfile.TemporaryDirectory() as tmpdir:
        log.debug("using tempdir %s", tmpdir)

        mime_message = EmailMessage()

        content = args.message + "\n\n"
        for attachment in attachments:
            report = process_attachment(
                attachment, message_id, tmpdir, mime_message, service
            )
            content = content + attachment.get("filename") + "\n" + str(report) + "\n"

        mime_message["To"] = headers["From"]
        mime_message["From"] = headers["To"]
        mime_message["Subject"] = "Re: " + headers["Subject"]
        mime_message["References"] = headers["Message-ID"]
        mime_message["In-Reply-To"] = headers["Message-ID"]

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <title>Email Response</title>
        <style>
            body {{font-family: Arial, sans-serif; color: #333;}}
            .notice {{margin-top: 20px; font-size: 12px; color: #777;}}
        </style>
        </head>
        <body>
        <div class="notice">
            <p>This is an automated response.</p>
        </div>
        <div class="content">
             <p>{content}</p>
        </div>
        </body>
        </html>
        """

        mime_message.attach(MIMEText(html, "html"))

        encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()

        log.debug("sending reply")
        _ = (
            service.users()
            .messages()
            .send(userId="me", body={"raw": encoded_message})
            .execute()
        )
        log.debug("sent reply")

        if not args.skip:
            log.debug("marking message %s as read", message_id)
            service.users().messages().modify(
                userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}
            ).execute()
            log.debug("marked message %s as read", message_id)


def process_data(data):
    """Process data."""

    original = data.copy()

    log.debug("processing data")
    data, target = split_data(data)
    classifier, report = predict(preprocess(data), target)
    log.debug("processed data")

    log.debug("predicting data")
    original["prediction"] = classifier.predict(data)
    log.debug("predicted data")

    return original, report


def process_attachment(attachment, message_id, tmpdir, mime_message, service):
    """Process attachment."""
    att_id = attachment["body"]["attachmentId"]

    attachment_filename = attachment["filename"]
    log.debug("processing attachment %s", attachment_filename)

    if ".csv" not in attachment_filename and ".xlsx" not in attachment_filename:
        log.debug("not a dataset")
        return

    att = (
        service.users()
        .messages()
        .attachments()
        .get(userId="me", messageId=message_id, id=att_id)
        .execute()
    )
    attachment_data = base64.urlsafe_b64decode(att["data"].encode("utf8"))

    if len(attachment_data) == 0:
        log.debug("attachment data is empty")
        return

    attachment_path = os.path.join(tmpdir, attachment_filename)
    log.debug(
        "saving attachment %s having %d bytes", attachment_path, len(attachment_data)
    )

    with open(attachment_path, "wb") as file:
        file.write(attachment_data)

    data = read_data(attachment_path)

    if data.empty:
        log.debug("dataframe data is empty")
        return

    data, report = process_data(data)

    if attachment_filename.endswith(".csv"):
        data.to_csv(attachment_filename)
    if attachment_filename.endswith(".xlsx"):
        data.to_excel(attachment_filename)

    with open(attachment_filename, "rb") as file:
        attachment_data = file.read()

    maintype, subtype = attachment["mimeType"].split("/")
    mime_message.add_attachment(
        attachment_data,
        maintype=maintype,
        subtype=subtype,
        filename=attachment_filename,
    )

    return report


def read_data(file):
    """Read input CSV/Excel directly or attached in EML as a Pandas DataFrame."""

    log.debug("reading input from %s", file)

    data = None

    if ".eml" in file:
        with open(file, "rb") as file_handle:
            message = email.message_from_binary_file(file_handle, policy=policy.default)
            with tempfile.TemporaryDirectory() as tmpdir:
                log.debug("using tempdir %s", tmpdir)
                for attachment in message.iter_attachments():
                    file = os.path.join(tmpdir, attachment.get_filename())
                    log.debug("saving attachment %s", file)
                    with open(file, "xb") as output_file:
                        output_file.write(attachment.get_payload(decode=True))
                        data = pandas.read_csv(file, encoding_errors="ignore")
                    break
    elif file.endswith(".csv.bz2") or file.endswith(".csv"):
        data = pandas.read_csv(file, encoding_errors="ignore")
    elif file.endswith(".xlsx.bz2") or file.endswith(".xlsx"):
        data = pandas.read_excel(file)

    log.debug("read %s rows and %s columns", len(data), len(data.columns))
    log.debug("read input")

    return data


def fill_missing_values(data: pandas.DataFrame):
    """Fill missing values using most frequent one."""
    log.debug("filling missing values")

    missing_cols = data.columns[data.isna().any()]
    if missing_cols.empty:
        log.debug("no columns with missing values")
        return

    most_frequent = data[missing_cols].mode().iloc[0]
    log.debug("filling %s with %s", missing_cols, list(most_frequent))
    data[missing_cols] = data[missing_cols].fillna(most_frequent)

    log.debug("filled %d missing values", len(missing_cols))


def convert_date_columns(data: pandas.DataFrame):
    """Convert date columns."""
    log.debug("converting date columns")
    cols = 0
    for column in data.select_dtypes(include=["object"]).columns:
        try:
            parsed = data[column].apply(lambda x: parser.parse(x, fuzzy=True))

            data[column + "_year"] = parsed.dt.year
            data[column + "_month"] = parsed.dt.month
            data[column + "_day"] = parsed.dt.day
            data[column + "_hour"] = parsed.dt.hour

            data[column + "_hour_sin"] = numpy.sin(
                2 * numpy.pi * data[column + "_hour"] / 24
            )
            data[column + "_hour_cos"] = numpy.cos(
                2 * numpy.pi * data[column + "_hour"] / 24
            )

            data[column + "_minute"] = parsed.dt.minute
            data[column + "_weekday"] = parsed.dt.weekday
            data[column + "_quarter"] = parsed.dt.quarter

            holidays = USFederalHolidayCalendar().holidays(
                start=parsed.min(), end=parsed.max()
            )
            data[column + "_holiday"] = parsed.isin(holidays)
            data[column + "_since"] = abs((parsed - datetime.now()).dt.days)
            data.drop(columns=column, inplace=True)
            log.debug("converted %s", column)
            cols = cols + 1
        except (AttributeError, ValueError):
            continue
    log.debug("converted %s date columns", cols)


def remove_constant_columns(data: pandas.DataFrame):
    """Remove constant columns."""
    log.debug("removing constant columns")

    nunique_counts = data.nunique()
    constant_cols = nunique_counts[nunique_counts == 1].index
    none_cols = data.columns[data.isna().all()]
    columns = constant_cols.union(none_cols)
    log.debug("removing constant columns: %s", list(columns))

    data.drop(columns=columns, inplace=True)

    log.debug("removed %d columns", len(columns))


def encode_categorical_columns(data: pandas.DataFrame):
    """Encode categorical columns."""

    log.debug("encoding categorical columns")

    categorical_columns = data.select_dtypes(include=["object"]).columns

    columns = []
    for col in categorical_columns:
        if (data[col].apply(lambda x: len(x.split())) < 3).all():
            if data[col].nunique() < 10:
                columns.append(col)
                log.debug("encode %s column", col)

    encoder = OneHotEncoder(handle_unknown="ignore", max_categories=8)
    encoded = pandas.DataFrame(
        encoder.fit_transform(data[columns]).toarray(),
        columns=encoder.get_feature_names_out(),
    )
    data[encoded.columns] = encoded
    log.debug("encoded %s categorical columns", len(columns))


def vectorize_text_columns(data: pandas.DataFrame):
    """Vectorize text columns."""
    log.debug("vectorizing text columns")
    categorical_columns = data.select_dtypes(include=["object"]).columns
    cols = 0
    for col in categorical_columns:
        if data[col].apply(lambda x: len(x.split())).mean() > 2:
            data[col + "_wc"] = data[col].apply(lambda x: len(x.split()))
            try:
                vectorizer = CountVectorizer(
                    stop_words="english",
                    max_features=8,
                    min_df=2,
                    max_df=0.9,
                    strip_accents="ascii",
                )
                matrix = vectorizer.fit_transform(data[col])
                vector = pandas.DataFrame(
                    matrix.toarray(),
                    columns=col + "_" + vectorizer.get_feature_names_out(),
                )
                data[vector.columns] = vector
                log.debug("vector %s", col)
                cols = cols + 1
            except ValueError:
                continue
        if data[col].apply(lambda x: len(x.split())).mean() > 4:
            data[col + "_polarity"] = data[col].apply(
                lambda x: TextBlob(x).sentiment.polarity
            )
            data[col + "_subjectivity"] = data[col].apply(
                lambda x: TextBlob(x).sentiment.subjectivity
            )
            data.drop(data.columns[data.nunique() == 1], axis=1, inplace=True)
    log.debug("vectorized %s text columns", cols)


def drop_categorical_columns(data: pandas.DataFrame):
    """Drop categorical columns in place."""
    log.debug("dropping categorical columns")
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns
    if len(categorical_columns) == 0:
        log.debug("no categorical columns to drop")
        return

    log.debug("drop columns: %s", categorical_columns.to_list())
    data.drop(columns=categorical_columns, inplace=True)
    log.debug("dropped %s categorical columns", len(categorical_columns))


def scale_numerical_columns(data: pandas.DataFrame):
    """Scale numerical columns."""
    log.debug("scaling numerical columns")
    numerical_columns = data.select_dtypes(include=["number"]).columns
    if len(numerical_columns) == 0:
        log.debug("no numerical columns to scale")
        return
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[numerical_columns])
    data[numerical_columns] = pandas.DataFrame(scaled, columns=numerical_columns)
    log.debug("scaled %s numerical columns", len(numerical_columns))


def remove_correlated_columns(data: pandas.DataFrame):
    """Remove correlated columns."""
    log.debug("dropping correlated columns")
    if len(data.columns) < 10:
        log.debug("not enough columns to remove correlated ones")
        return

    correlation_matrix = data.corr().abs()
    upper = correlation_matrix.where(
        numpy.triu(numpy.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
    log.debug("drop correlated columns: %s", to_drop)
    data.drop(columns=to_drop, inplace=True)

    log.debug("dropped correlated columns")


def preprocess(data: pandas.DataFrame) -> pandas.DataFrame:
    """Preprocess data."""
    log.debug("preprocessing")

    with io.StringIO() as buffer:
        data.info(buf=buffer)

    fill_missing_values(data)
    convert_date_columns(data)
    remove_constant_columns(data)
    encode_categorical_columns(data)
    vectorize_text_columns(data)
    drop_categorical_columns(data)
    scale_numerical_columns(data)
    remove_correlated_columns(data)

    log.debug("preprocessed")

    return data


def predict(
    data: pandas.DataFrame, label: pandas.DataFrame
) -> tuple[float, typing.Any | None]:
    """Predict label."""

    classifiers = {
        "SVM": SVC(kernel="linear", random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        # "LogisticRegression": LogisticRegression(random_state=42),
    }

    param_grids: typing.Dict[str, typing.Dict[str, typing.Any]] = {
        "SVM": {},
        "RandomForest": {"min_samples_split": [2]},
        # "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        # "RandomForest": {
        #     "n_estimators": [100, 200],
        #     "max_depth": [None, 10, 20],
        #     "min_samples_split": [2, 5],
        # },
        # "LogisticRegression": {"C": [0.1, 1, 10], "penalty": ["l2"]},
    }

    log.debug("predicting using %s classifiers", list(classifiers.keys()))
    log.debug("predict using %d columns %s", len(data.columns), data.columns.to_list())
    log.debug("predict using %s label", label.columns.to_list())

    log.debug("training with 0.25 split")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        data, label, test_size=0.25, random_state=42
    )

    best_classifier = None
    best_report = None

    for name, classifier in classifiers.items():
        log.debug("tuning hyperparameters for %s", name)
        grid_search = GridSearchCV(
            estimator=classifier, param_grid=param_grids[name], cv=5, n_jobs=-1
        )
        grid_search.fit(x_train, y_train.values.ravel())
        log.debug("%s best parameters: %s", name, grid_search.best_params_)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        log.debug("%s model got %s accuracy", name, round(report["accuracy"], 5))
        if best_report is None or best_report["accuracy"] < report["accuracy"]:
            best_classifier = best_model
            best_report = report

    return best_classifier, best_report


def main(args) -> None:
    """Poll for emails to process and respond."""
    process_emails(args)


def parse_arguments():
    """Parse arguments."""

    parser = argparse.ArgumentParser(
        description="Script that answers predictions of emails with dataset attachments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-q",
        "--query",
        default="from:me subject:analyze is:unread has:attachment",
        help="Gmail filter query",
    )
    parser.add_argument("-l", "--loglevel", default="DEBUG", help="Logging level")
    parser.add_argument(
        "-c", "--credentials", default="credentials.json", help="Gmail credentials"
    )
    parser.add_argument("-t", "--token", default="token.json", help="Gmail token")
    parser.add_argument(
        "-m",
        "--message",
        default="<p>Thanks, attached is your dataset prediction.</p>",
        help="Reply message",
    )
    parser.add_argument(
        "-s",
        "--skip",
        action="store_true",
        default=False,
        help="Do not mark as read processed messages.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    log.setLevel(logging.DEBUG)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=getattr(logging, args.loglevel.upper(), None),
    )

    main(args)

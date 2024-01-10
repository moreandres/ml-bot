#!/usr/bin/env python3

"""Script that answers predictions of emails with dataset attachments."""

import logging
import os
import base64
import tempfile

from datetime import datetime

from email.message import EmailMessage
import email
from email import policy

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

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")


def get_gmail_credentials() -> Credentials:
    """
    Get Gmail credentials.
    Reuse, refresh or get from scratch as required.
    Use working directory to read credentials and store token JSON files.
    """

    auth_scopes = ["https://www.googleapis.com/auth/gmail.modify"]

    credentials = None

    log.debug("getting credentials")

    if os.path.exists("token.json"):
        log.debug("reusing token")
        credentials = Credentials.from_authorized_user_file("token.json", auth_scopes)
        if credentials.valid:
            return credentials

        log.debug("refreshing token")
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
    else:
        log.debug("getting new token")
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", auth_scopes
        )
        credentials = flow.run_local_server(port=0)

    log.debug("writing token")
    with open("token.json", "w", encoding="utf8") as token:
        token.write(credentials.to_json())

    return credentials


def process_attachment(attachment):
    """Process single attachment."""
    input_data = read_data(attachment)
    target = get_target(input_data)
    target_data = input_data[target]
    raw_data = input_data.drop(columns=target)
    processed_data = preprocess(raw_data)
    input_data["prediction"] = predict(processed_data, target_data)


def process_emails():
    """Process emails."""

    service = build("gmail", "v1", credentials=get_gmail_credentials())

    log.debug("searching messages")
    results = (
        service.users()
        .messages()
        .list(userId="me", q="from:me subject:analyze has:attachment")
        .execute()
    )

    messages = results.get("messages", [])
    log.debug("processing %d messages", len(messages))

    for message in messages:
        message_id = message["id"]
        log.debug("processing message %s", message_id)

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

        mime_message = EmailMessage()
        mime_message["To"] = headers["From"]
        mime_message["From"] = headers["To"]
        mime_message["Subject"] = "Re: " + headers["Subject"]
        mime_message["References"] = headers["Message-ID"]
        mime_message["In-Reply-To"] = headers["Message-ID"]

        mime_message.set_content(
            "Hi, this is automated response with an updated attachment including predictions."
        )

        log.debug("processing %s attachments", len(attachments))

        with tempfile.TemporaryDirectory() as tmpdir:
            log.debug("using tempdir %s", tmpdir)

            for attachment in attachments:
                att_id = attachment["body"]["attachmentId"]

                attachment_filename = attachment["filename"]
                log.debug("processing attachment %s", attachment_filename)

                att = (
                    service.users()
                    .messages()
                    .attachments()
                    .get(userId="me", messageId=message_id, id=att_id)
                    .execute()
                )
                attachment_data = base64.urlsafe_b64decode(att["data"].encode("utf8"))

                log.debug("saving attachment")
                attachment_path = os.path.join(tmpdir, attachment_filename)
                with open(attachment_path, "wb") as f:
                    f.write(attachment_data)

                    maintype, subtype = attachment["mimeType"].split("/")

                    with open(attachment_filename, "rb") as fp:
                        attachment_data = fp.read()
                    mime_message.add_attachment(
                        attachment_data,
                        maintype=maintype,
                        subtype=subtype,
                        filename=attachment_filename,
                    )

            encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()

            _ = (
                service.users()
                .messages()
                .send(userId="me", body={"raw": encoded_message})
                .execute()
            )

            log.debug("sending reply")


def read_data(file):
    """Read input CSV as data frame."""

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
                        data = pandas.read_csv(file)
                    break
    else:
        data = pandas.read_csv(file)

    log.debug("read %s rows", len(data))
    log.debug("read %s columns", len(data.columns))
    log.debug("read input")

    return data


def fill_missing_values(data: pandas.DataFrame):
    """Fill missing values using most frequent one."""
    log.debug("filling missing values")
    for col in data.columns:
        if data[col].isna().any():
            most_frequent = data[col].mode()[0]
            log.debug("filling %s column with %s", col, most_frequent)
            data[col].fillna(most_frequent, inplace=True)
    log.debug("filled missing values")


def convert_date_columns(data: pandas.DataFrame):
    """Convert date columns."""
    log.debug("converting date columns")
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
        except ValueError:
            continue
    log.debug("converted date columns")


def remove_constant_columns(data: pandas.DataFrame):
    """Remove constant columns."""
    log.debug("removing constant columns")
    for col in data.columns:
        if data[col].nunique() == 1:
            log.debug("remove %s", col)
    data.drop(data.columns[data.nunique() == 1], axis=1, inplace=True)
    log.debug("removed constant columns")


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
    log.debug("encoded categorical columns")


def vectorize_text_columns(data: pandas.DataFrame):
    """Vectorize text columns."""
    log.debug("vectorizing text columns")
    categorical_columns = data.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        data[col + "_wc"] = data[col].apply(lambda x: len(x.split()))
        if data[col].apply(lambda x: len(x.split())).mean() > 2:
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
                    matrix.toarray(), columns=vectorizer.get_feature_names_out()
                )
                data[vector.columns] = vector
                log.debug("vector %s", col)
            except ValueError:
                continue
            data[col + "_sentiment"] = data[col].apply(
                lambda x: TextBlob(x).sentiment.polarity
            )
            data.drop(data.columns[data.nunique() == 1], axis=1, inplace=True)
    log.debug("vectorized text columns")


def drop_categorical_columns(data: pandas.DataFrame):
    """Drop categorical columns in place."""
    log.debug("dropping categorical columns")
    categorical_columns = data.select_dtypes(include=["object"]).columns
    log.debug(
        "drop %s columns: %s", len(categorical_columns), categorical_columns.to_list()
    )
    data.drop(columns=categorical_columns, inplace=True)
    log.debug("dropped categorical columns")


def scale_numerical_columns(data: pandas.DataFrame):
    """Scale numerical columns."""
    log.debug("scaling numerical columns")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    data[data.columns] = pandas.DataFrame(scaled, columns=data.columns)
    log.debug("scaled numerical columns")


def preprocess(data: pandas.DataFrame) -> pandas.DataFrame:
    """Preprocess data."""
    log.debug("preprocessing")

    fill_missing_values(data)
    convert_date_columns(data)
    remove_constant_columns(data)
    encode_categorical_columns(data)
    vectorize_text_columns(data)
    drop_categorical_columns(data)
    scale_numerical_columns(data)

    log.debug("preprocessed")

    return data


def predict(data: pandas.DataFrame, label: pandas.DataFrame) -> float:
    """Predict label."""

    log.debug("predicting")

    log.debug("predict using %s columns: %s", len(data.columns), data.columns.to_list())

    log.debug("training")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        data, label, random_state=42
    )

    classifier = SVC(kernel="linear", random_state=42)
    classifier.fit(x_train, y_train)
    log.debug("trained")

    log.debug("testing")
    y_pred = classifier.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    log.debug("classification report: %s", report)
    log.debug("tested")

    log.debug("predicted")
    return report["accuracy"]


def get_target(data: pandas.DataFrame) -> str:
    """
    Get name of target column feature.
    Check if usual names are there, otherwise fallback to last column.
    """

    names = ["target"]

    for name in names:
        if name in data.columns:
            return name

    return data.columns[-1]


def main() -> None:
    """Poll for emails to process and respond."""
    process_emails()


if __name__ == "__main__":
    main()

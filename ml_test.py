"""UT coverage for ml.py."""

import pandas
import pytest

import ml


@pytest.mark.parametrize(
    "inpt,output",
    [
        (
            {
                "label": [0.0, 1.0, 0.0],
                "constant_categorical": ["true", "true", "true"],
                "constant_numerical": [1.0, 1.0, 1.0],
                "feature_numerical": [1.0, 0, 1.0],
                "feature_categorical": ["good", "bad", "good"],
                "feature_dates": [
                    "2023-10-20T22:05:37.207919",
                    "2022-10-20T22:05:37.207919",
                    "2023-10-20T22:05:37.207919",
                ],
                "missing_numerical": [1.0, 1.0, None],
                "missing_categorical": ["good", "good", None],
                "feature_text": [
                    "this is hello",
                    "this is also hello",
                    "this is not",
                ],
            },
            {
                "label": [0.0, 1.0, 0.0],
                "feature_numerical": [1.0, 0.0, 1.0],
                "feature_dates_year": [1.0, 0.0, 1.0],
                "feature_dates_weekday": [1.0, 0.0, 1.0],
                "feature_dates_since": [0.0, 1.0, 0.0],
                "feature_categorical_bad": [0.0, 1.0, 0.0],
                "feature_categorical_good": [1.0, 0.0, 1.0],
                "feature_text_wc": [0.0, 1.0, 0.0],
                "hello": [1.0, 1.0, 0.0],
            },
        )
    ],
)
def test_preprocess(inpt, output):
    """UT coverage for preprocess() with valid data."""
    expected = pandas.DataFrame(output)
    actual = ml.preprocess(pandas.DataFrame(inpt))
    assert expected.equals(actual), "actual should match expected"


@pytest.mark.parametrize(
    "inpt,label,accuracy",
    [
        (
            {
                "feature_numerical": [1.0, 0.0, 1.0],
            },
            {
                "label": [0.0, 1.0, 0.0],
            },
            1.0,
        )
    ],
)
def test_predict(inpt, label, accuracy):
    """Test predict()."""
    assert (
        ml.predict(pandas.DataFrame(inpt), pandas.DataFrame(label)) == accuracy
    ), "actual should match expected"


@pytest.mark.parametrize(
    "inpt,label,accuracy",
    [
        ("t/abalone.csv.bz2", "rings", 0.2574162679425837),
        ("t/ad.csv.bz2", "class", 0.9707317073170731),
        ("t/adult.csv.bz2", "income", 0.856359020555237),
        ("t/bank.csv.bz2", "y", 0.8997766339710596),
        ("t/cancer.csv.bz2", "diagnosis", 0.986013986013986),
        ("t/car.csv.bz2", "class", 0.9236111111111112),
        ("t/heart.csv.bz2", "num", 0.5478260869565217),
        ("t/iris.csv.bz2", "target", 1.0),
        ("t/mushroom.csv.bz2", "class", 1.0),
        ("t/titanic.csv.bz2", "Survived", 0.8295964125560538),
        ("t/titanic.eml", "Survived", 0.8295964125560538),
    ],
)
def test_datasets(inpt, label, accuracy):
    """Test well-known datasets."""
    data = ml.read_data(inpt)
    target = data[label]
    data.drop(columns=label, inplace=True)

    assert (
        ml.predict(ml.preprocess(data), target) == accuracy
    ), "actual should match expected"


def test_read_data():
    """UT coverage for read_data()."""
    assert len(ml.read_data("t/iris.csv.bz2")) == 150, "should read data"

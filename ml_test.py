"""UT coverage for ml.py."""

import pandas
import pytest

import ml


@pytest.mark.parametrize(
    "inpt,data,target",
    [
        (
            {
                "label": [0.0, 1.0, 0.0],
                "feature_numerical": [1.0, 0.0, 1.0],
            },
            {
                "feature_numerical": [1.0, 0.0, 1.0],
            },
            {
                "label": [0.0, 1.0, 0.0],
            },
        )
    ],
)
def test_split_data(inpt, data, target):
    """UT coverage for split_data()."""
    actual_data, actual_target = ml.split_data(pandas.DataFrame(inpt))
    print(actual_data)
    print(actual_target)
    pandas.testing.assert_frame_equal(actual_data, pandas.DataFrame(data))
    pandas.testing.assert_frame_equal(actual_target, pandas.DataFrame(target))


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
                "feature_text_hello": [1.0, 1.0, 0.0],
            },
        )
    ],
)
def test_preprocess(inpt, output):
    """UT coverage for preprocess() with valid data."""
    expected = pandas.DataFrame(output)
    actual = ml.preprocess(pandas.DataFrame(inpt))
    pandas.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "inpt,label,expected",
    [
        (
            {
                "feature_numerical": [
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                ],
            },
            {
                "label": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            },
            1.0,
        )
    ],
)
def test_predict(inpt, label, expected):
    """Test predict()."""
    _, report = ml.predict(pandas.DataFrame(inpt), pandas.DataFrame(label))
    assert report["accuracy"] == expected, "actual accuracy should match expected"


@pytest.mark.parametrize(
    "inpt,label,expected",
    [
        (
            "t/used.csv.bz2",
            "price",
            0.94,
        ),
        (
            "t/life.csv.bz2",
            "Life expectancy ",
            0.97,
        ),
        (
            "t/insurance.csv.bz2",
            "charges",
            0.85,
        ),
        (
            "t/loan.csv.bz2",
            "loan_status",
            0.93,
        ),
        (
            "t/behavior.csv.bz2",
            "User Behavior Class",
            1.0,
        ),
        (
            "t/mediaciones.csv.bz2",
            "reapertura",
            0.94,
        ),
        (
            "t/abalone.csv.bz2",
            "rings",
            0.53,
        ),  # Abalone Age Prediction Using Machine Learning
        ("t/ad.csv.bz2", "class", 0.96),
        ("t/soccer.csv.bz2", "Target", 0.87),
        (
            "t/adult.csv.bz2",
            "income",
            0.85,
        ),  # https://www.cs.toronto.edu/~delve/data/adult/desc.html
        (
            "t/bank.csv.bz2",
            "y",
            0.95,
        ),  # https://archive.ics.uci.edu/dataset/222/bank+marketing
        ("t/cancer.csv.bz2", "diagnosis", 0.99),
        ("t/car.csv.bz2", "class", 0.97),
        (
            "t/defects.csv.bz2",
            "DefectStatus",
            0.96,
        ),  # https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset
        (
            "t/fire.csv.bz2",
            "STATUS",
            0.97,
        ),  # https://www.kaggle.com/datasets/muratkokludataset/acoustic-extinguisher-fire-dataset
        (
            "t/flights.csv.bz2",
            "satisfaction_v2",
            1.0,
        ),  # https://www.kaggle.com/datasets/johndddddd/customer-satisfaction
        (
            "t/fruit.csv.bz2",
            "Class",
            0.92,
        ),  # https://www.kaggle.com/datasets/muratkokludataset/date-fruit-datasets
        (
            "t/gaming.csv.bz2",
            "EngagementLevel",
            0.89,
        ),  # https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset
        ("t/heart.csv.bz2", "num", 0.63),
        (
            "t/hepatitis.csv.bz2",
            "Category",
            0.96,
        ),  # https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset
        (
            "t/iris.csv.bz2",
            "target",
            1.0,
        ),  # https://archive.ics.uci.edu/dataset/53/iris
        (
            "t/mushroom.csv.bz2",
            "class",
            1.0,
        ),  # https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset
        (
            "t/pistachio.csv.bz2",
            "Class",
            0.92,
        ),  # https://www.kaggle.com/datasets/muratkokludataset/pistachio-dataset
        (
            "t/pumpkin.csv.bz2",
            "Class",
            0.88,
        ),  # https://www.kaggle.com/datasets/muratkokludataset/pumpkin-seeds-dataset
        (
            "t/stars.csv.bz2",
            "class",
            0.97,
        ),  # https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
        (
            "t/stroke.csv.bz2",
            "stroke",
            0.94,
        ),  # https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
        (
            "t/students.csv.bz2",
            "GradeClass",
            0.90,
        ),  # https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset
        (
            "t/titanic.csv.bz2",
            "Survived",
            0.85,
        ),  # Will Cukierski. (2012). Titanic - Machine Learning from Disaster. Kaggle. https://kaggle.com/competitions/titanic
        ("t/titanic.eml", "Survived", 0.84753),
        ("t/titanic.xlsx", "Survived", 0.84753),
        (
            "t/water.csv.bz2",
            "Potability",
            0.68,
        ),  # https://www.kaggle.com/datasets/adityakadiwal/water-potability
        (
            "t/weather.csv.bz2",
            "RainTomorrow",
            0.84,
        ),  # https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package
        (
            "t/purchase.csv.bz2",
            "PurchaseStatus",
            0.95,
        ),  # https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset/data
        (
            "t/recruit.csv.bz2",
            "HiringDecision",
            0.93,
        ),  # https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data
        ("t/enron.csv.bz2", "label", 0.88551),
        ("t/sms.csv.bz2", "v1", 0.91),
    ],
)
def test_classification_datasets(inpt, label, expected):
    """Test well-known datasets."""
    data = ml.read_data(inpt)
    target = data[[label]]
    data.drop(columns=label, inplace=True)

    _, report = ml.predict(ml.preprocess(data), target)
    assert round(report["accuracy"], 2) == round(
        expected, 2
    ), f"actual should match expected for {label}"


@pytest.mark.parametrize(
    "inpt,label,expected",
    [
        ("t/bikes.csv.bz2", "Rented Bike Count", 0.92),
        ("t/houses.csv.bz2", "price", 0.87),
        ("t/walmart.csv.bz2", "Weekly_Sales", 0.97),
        ("t/seats.csv.bz2", "pasajeros", 0.97),
    ],
)
def test_regression_datasets(inpt, label, expected):
    """Test well-known datasets."""
    data = ml.read_data(inpt)
    target = data[[label]]
    data.drop(columns=label, inplace=True)

    _, report = ml.predict(ml.preprocess(data), target)
    assert round(report["accuracy"], 2) == round(
        expected, 2
    ), f"actual should match expected for {label}"


def test_read_data():
    """UT coverage for read_data()."""
    assert (
        len(ml.read_data("t/iris.csv.bz2")) == 150
    ), "should read data from compressed CSV"
    assert len(ml.read_data("t/titanic.eml")) == 891, "should read data from EML email"


def test_fill_missing_values_no_missing():
    """Test that no changes are made when there are no missing values."""
    data = pandas.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    original_df = data.copy()

    ml.fill_missing_values(data)

    pandas.testing.assert_frame_equal(original_df, data)


def test_fill_missing_values_single_column():
    """Test filling missing values in a single column."""
    data = pandas.DataFrame({"A": [1.0, 2.0, None, 2.0], "B": [4.0, 5.0, 6.0, 7.0]})

    ml.fill_missing_values(data)

    expected_df = pandas.DataFrame(
        {
            "A": [1.0, 2.0, 2.0, 2.0],
            "B": [4.0, 5.0, 6.0, 7.0],
        }  # 2 is the most frequent value
    )

    pandas.testing.assert_frame_equal(expected_df, data)


def test_fill_missing_values_multiple_columns():
    """Test filling missing values in multiple columns."""
    data = pandas.DataFrame({"A": [1.0, None, 3.0, None], "B": [None, 5.0, 5.0, 6.0]})

    ml.fill_missing_values(data)

    expected_df = pandas.DataFrame(
        {
            "A": [1.0, 1.0, 3.0, 1.0],  # 1 is the most frequent value
            "B": [5.0, 5.0, 5.0, 6.0],  # 5 is the most frequent value
        }
    )

    pandas.testing.assert_frame_equal(expected_df, data)


def test_fill_missing_values_single_value():
    """Test filling a column that has only a single value."""
    data = pandas.DataFrame({"A": [1.0, None, None], "B": [1.0, 2.0, 3.0]})

    ml.fill_missing_values(data)

    expected_df = pandas.DataFrame(
        {"A": [1.0, 1.0, 1.0], "B": [1.0, 2.0, 3.0]}  # mode is single value
    )

    pandas.testing.assert_frame_equal(expected_df, data)


def test_fill_missing_values_multiple_modes():
    """Test filling values when there are multiple modes."""
    data = pandas.DataFrame(
        {"A": [1.0, 2.0, 2.0, 2.0, None], "B": [3.0, None, 4.0, 4.0, 4.0]}
    )

    ml.fill_missing_values(data)

    expected_df = pandas.DataFrame(
        {
            "A": [
                1.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ],  # 1 and 2 are modes, but 2 is selected
            "B": [
                3.0,
                4.0,
                4.0,
                4.0,
                4.0,
            ],  # 3 and 4 are modes, but 4 is selected
        }
    )

    pandas.testing.assert_frame_equal(expected_df, data)


def test_remove_constant_columns_no_constant():
    """Test that no columns are removed when there are no constant columns."""
    data = pandas.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0], "C": [7.0, 8.0, 9.0]}
    )
    original_df = data.copy()

    ml.remove_constant_columns(data)

    pandas.testing.assert_frame_equal(original_df, data)


def test_remove_constant_columns_one_constant():
    """Test that one constant column is removed."""
    data = pandas.DataFrame(
        {"A": [1.0, 1.0, 1.0], "B": [2.0, 3.0, 4.0], "C": [5.0, 5.0, 5.0]}
    )

    ml.remove_constant_columns(data)

    expected_df = pandas.DataFrame({"B": [2.0, 3.0, 4.0]})

    pandas.testing.assert_frame_equal(expected_df, data)


def test_remove_constant_columns_multiple_constants():
    """Test that multiple constant columns are removed."""
    data = pandas.DataFrame(
        {
            "A": [1.0, 1.0, 1.0],
            "B": [2.0, 2.0, 2.0],
            "C": [3.0, 4.0, 5.0],
            "D": [7.0, 7.0, 7.0],
        }
    )

    ml.remove_constant_columns(data)

    expected_df = pandas.DataFrame({"C": [3.0, 4.0, 5.0]})

    pandas.testing.assert_frame_equal(expected_df, data)


def test_remove_constant_columns_all_constant():
    """Test that all columns are removed when they are all constant."""
    data = pandas.DataFrame({"A": [1.0, 1.0, 1.0], "B": [2.0, 2.0, 2.0]})

    ml.remove_constant_columns(data)

    assert data.empty, "data should be empty"


def test_remove_constant_columns_empty():
    """Test behavior on an empty DataFrame."""
    data = pandas.DataFrame(columns=["A", "B"])

    ml.remove_constant_columns(data)

    assert data.empty, "data should be empty"

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
from utils.config_loader import load_parameters
from config import STATUS_MAP

# Configure joblib to use less memory
joblib.parallel.BACKEND = 'loky'

def create_pipeline(params):
    """Create the machine learning pipeline for job matching.

    Args:
        params (dict): Dictionary of parameters for the pipeline

    Returns:
        Pipeline: Configured scikit-learn pipeline
    """
    preprocessor = ColumnTransformer([
        ("tfidf_ativ", TfidfVectorizer(
            max_features=params['TFIDF_JOB_DESCRIPTION_MAX_FEATURES'],
            ngram_range=params['TFIDF_JOB_DESCRIPTION_NGRAM_RANGE'],
            strip_accents="unicode",
            dtype=np.float32,
            min_df=params['TFIDF_MIN_DF'],
            max_df=params['TFIDF_MAX_DF']
        ), "job_description"),
        ("tfidf_comp", TfidfVectorizer(
            max_features=params['TFIDF_JOB_REQUIREMENTS_MAX_FEATURES'],
            ngram_range=params['TFIDF_JOB_REQUIREMENTS_NGRAM_RANGE'],
            strip_accents="unicode",
            dtype=np.float32,
            min_df=params['TFIDF_MIN_DF'],
            max_df=params['TFIDF_MAX_DF']
        ), "job_requirements"),
        ("tfidf_cv", TfidfVectorizer(
            max_features=params['TFIDF_CANDIDATE_CV_MAX_FEATURES'],
            ngram_range=params['TFIDF_CANDIDATE_CV_NGRAM_RANGE'],
            strip_accents="unicode",
            dtype=np.float32,
            min_df=params['TFIDF_MIN_DF'],
            max_df=params['TFIDF_MAX_DF']
        ), "candidate_cv"),
    ], remainder="drop")

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            max_iter=params['LOGISTIC_REGRESSION_MAX_ITER'],
            n_jobs=1,
            solver='saga',
            tol=params['LOGISTIC_REGRESSION_TOL']
        ))
    ])

    return pipeline

def train_model(df):
    """Train the job matching model using the configured pipeline and parameters.

    Returns:
        tuple: (auc, grid, X_test, y_test) containing the AUC score, trained grid search object,
               and test data for additional metrics
    """
    print("\n=== Starting Training Process ===")
    print("Loading parameters...")
    params = load_parameters()
    print("Parameters loaded successfully")

    print("\n=== Loading and Processing Data ===")
    print(f"Total samples in dataset: {len(df)}")

    print("\n=== Preparing Features and Target ===")
    x = df[["job_description", "job_requirements", "candidate_cv"]]
    y = df["status"].map(STATUS_MAP)
    print(f"Feature shape: {x.shape}")
    print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")

    print("\n=== Splitting Data ===")
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=params['TEST_SIZE'], random_state=params['RANDOM_STATE']
    )
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    print("\n=== Creating Pipeline ===")
    print("Initializing TF-IDF vectorizers...")
    pipeline = create_pipeline(params)
    print("Pipeline created successfully")

    print("\n=== Starting Grid Search ===")
    print(f"Testing C values: {params['GRID_SEARCH_C_VALUES']}")
    print(f"Performing {params['GRID_SEARCH_CV']}-fold cross-validation...")
    print(f"Using {params['GRID_SEARCH_N_JOBS']} parallel jobs")
    print("Progress will be shown below:")

    try:
        print("\nInitializing GridSearchCV...")
        grid = GridSearchCV(
            pipeline,
            {"clf__C": params['GRID_SEARCH_C_VALUES']},
            cv=params['GRID_SEARCH_CV'],
            scoring=params['GRID_SEARCH_SCORING'],
            n_jobs=params['GRID_SEARCH_N_JOBS'],
            verbose=2,  # Increased verbosity
            error_score='raise',
            pre_dispatch='n_jobs',
            return_train_score=False
        )

        print("\nStarting model training...")
        with joblib.parallel_backend('loky', n_jobs=params['GRID_SEARCH_N_JOBS']):
            grid.fit(X_train, y_train)

        print(f"\nTraining completed!")
        print(f"Best C value: {grid.best_params_['clf__C']}")
        print(f"Best cross-validation score: {grid.best_score_:.4f}")

        # Calculate AUC on test set
        y_pred = grid.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        print(f"AUC on test set: {auc:.4f}")

        return auc, grid, X_test, y_test

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


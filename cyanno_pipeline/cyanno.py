#!/usr/bin/env python3
"""
CyAnno Pipeline Module
======================

This module provides a reusable, simplified implementation of the CyAnno approach
for automated cytometry cell type annotation. It is designed to be imported by 
other scripts or workflow systems (e.g., Snakemake, Nextflow, etc.).

The key functionality provided here:
- Machine learning–based classification using Random Forest
- Optional data normalization
- Training with validation split
- Prediction with probability scores
- Evaluation (F1 scores, confusion matrix)
- Model saving/loading
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, cast
import warnings

# Suppress unnecessary warnings for cleaner workflow logs
warnings.filterwarnings('ignore')


class CyAnnoClassifier:
    """
    CyAnnoClassifier
    ----------------
    A simplified and reusable version of the CyAnno ML model.
    Designed to annotate cytometry cell populations based on markers.

    Attributes
    ----------
    markers : List[str]
        Feature names (column names) to use in training and prediction.

    normalize : bool
    classifier : RandomForestClassifier
        Underlying ML model used for classification.

    cell_types : List[str]
        Sorted list of unique cell type labels seen during training.

    is_trained : bool
        Flag indicating whether the model has been trained.
    """

    def __init__(self,
                 markers: List[str],
                 random_state: int = 42):
        """
        Constructor for CyAnnoClassifier.

        Parameters
        ----------
        markers : list of str
            Names of the marker columns to use.
        random_state : int
            Seed to ensure reproducibility.
        """

        # Store configuration
        self.markers = markers
        self.random_state = random_state

        # Main machine learning model
        self.classifier = RandomForestClassifier(
            n_estimators=100,   # Number of decision trees
            random_state=random_state,
            n_jobs=-1           # Use all CPU cores
        )

        self.cell_types = None
        self.is_trained = False

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


    # ----------------------------------------------------------------------
    # INTERNAL HELPER: DATA PREPROCESSING
    # ----------------------------------------------------------------------
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Internal method: Extracts marker columns without extra transformation.

        Parameters
        ----------
        data : pandas DataFrame

        Returns
        -------
        X : numpy.ndarray
            Preprocessed numerical matrix suitable for ML algorithms.
        """

        # select only the relevant marker columns
        marker_data = data[self.markers].copy()

        return marker_data.to_numpy()


    # ----------------------------------------------------------------------
    # TRAINING
    # ----------------------------------------------------------------------
    def train(self,
              handgated_data: pd.DataFrame,
              cell_type_column: str = 'cell_type') -> Dict:
        """
        Train the classifier using manually gated data.

        Parameters
        ----------
        handgated_data : pandas DataFrame
            Data frame containing marker intensities AND cell type labels.

        cell_type_column : str
            Name of column containing the known cell type labels.

        Returns
        -------
        metrics : dict
            Includes F1 score on validation split and model summary.
        """

        self.logger.info("Starting CyAnno training...")

        # Ensure that all required markers exist
        missing_markers = [m for m in self.markers if m not in handgated_data.columns]
        if missing_markers:
            raise ValueError(f"Missing markers in data: {missing_markers}")

        # Ensure label column is present
        if cell_type_column not in handgated_data.columns:
            raise ValueError(f"Cell type column '{cell_type_column}' not found")

        # pre-process features
        X = self._preprocess_data(handgated_data)

        # extract labels
        y = handgated_data[cell_type_column].values

        # store all observed cell types
        self.cell_types = sorted(list(set(y)))
        self.logger.info(f"Found {len(self.cell_types)} cell types: {self.cell_types}")

        # create train/validation split
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y   # ensures balanced representation per cell type
            )

        except ValueError:
            # Stratified split fails when some classes have <2 samples
            self.logger.warning("Stratification failed; splitting without stratification.")
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=None
            )

        # fit the RandomForest model
        self.classifier.fit(X_train, y_train)

        self.is_trained = True

        # evaluate on validation set
        y_pred = self.classifier.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='weighted')

        metrics = {
            'f1_score': f1,
            'n_training_samples': len(X_train),
            'n_validation_samples': len(X_val),
            'cell_types': self.cell_types,
            'classification_report': classification_report(y_val, y_pred)
        }

        self.logger.info(f"Training completed. Validation F1 Score: {f1:.3f}")
        return metrics


    # ----------------------------------------------------------------------
    # PREDICTION
    # ----------------------------------------------------------------------
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict cell types for new, unlabeled cytometry data.

        Parameters
        ----------
        data : pandas DataFrame

        Returns
        -------
        predictions : np.ndarray
            Predicted cell type labels.

        probabilities : np.ndarray
            Probabilities per class (rows = cells, columns = cell types).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")

        X = self._preprocess_data(data)

        predictions = cast(np.ndarray, self.classifier.predict(X))
        probabilities = cast(np.ndarray, self.classifier.predict_proba(X))

        return predictions, probabilities


    # ----------------------------------------------------------------------
    # MODEL EVALUATION (OPTIONAL)
    # ----------------------------------------------------------------------
    def evaluate(self,
                 test_data: pd.DataFrame,
                 true_labels_column: str = 'cell_type') -> Dict:
        """
        Evaluate the classifier on labeled test data.

        Parameters
        ----------
        test_data : pandas DataFrame
            Must include true labels and marker columns.

        true_labels_column : str
            Column with true cell type labels.

        Returns
        -------
        metrics : dict
        """

        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")

        predictions, probabilities = self.predict(test_data)
        true_labels = test_data[true_labels_column].values

        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_per_class = cast(
            np.ndarray,
            f1_score(true_labels, predictions, average=None),  # type: ignore[arg-type]
        )

        metrics = {
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_per_class': dict(zip(self.cell_types or [], f1_per_class.tolist())),
            'classification_report': classification_report(true_labels, predictions),
            'confusion_matrix': confusion_matrix(true_labels, predictions).tolist(),
            'n_test_samples': len(test_data)
        }

        self.logger.info(f"Evaluation completed. Weighted F1 Score: {f1_weighted:.3f}")
        return metrics


    # ----------------------------------------------------------------------
    #   MODEL SERIALIZATION
    # ----------------------------------------------------------------------
    def save_model(self, filepath: Path):
        """
        Save trained model + metadata to disk using pickle.
        """
        if not self.is_trained:
            raise ValueError("No trained model to save.")

        model_data = {
            'classifier': self.classifier,
            'markers': self.markers,
            'cell_types': self.cell_types,
            'random_state': self.random_state
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Path):
        """
        Load a previously saved model.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.classifier = model_data['classifier']
        self.markers = model_data['markers']
        self.cell_types = model_data['cell_types']
        self.random_state = model_data['random_state']

        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")
# End of CyAnnoClassifier

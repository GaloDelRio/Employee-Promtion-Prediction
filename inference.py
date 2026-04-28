from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from tensorflow import keras

from config import MODEL_PATH, PREPROCESSOR_PATH, THRESHOLD
from model import SliceColumn  # noqa: F401  # fuerza el registro de la capa custom
from preprocessing import load_preprocessor, transform_new_data
from utils import make_model_inputs


class PromotionPredictor:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        preprocessor_path: Path = PREPROCESSOR_PATH,
    ):
        self.preprocessor = load_preprocessor(preprocessor_path)
        self.threshold = THRESHOLD
        self.model = keras.models.load_model(model_path, compile=False)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        x_cat, x_num = transform_new_data(df, self.preprocessor)
        probabilities = self.model.predict(
            make_model_inputs(x_cat, x_num),
            batch_size=1024,
            verbose=0,
        ).reshape(-1)

        preds = (probabilities >= self.threshold).astype(int)

        result = df.copy()
        result['promoted_probability'] = probabilities.astype(float)
        result['promoted_prediction'] = preds.astype(int)
        return result

    def predict_one(self, input_data: Dict) -> Tuple[int, float]:
        df = pd.DataFrame([input_data])
        result = self.predict_dataframe(df)
        pred = int(result.loc[0, 'promoted_prediction'])
        proba = float(result.loc[0, 'promoted_probability'])
        return pred, proba
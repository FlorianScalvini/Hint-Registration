import torchio as tio
import pandas as pd
from typing import Callable, List

def subjects_from_csv(dataset_path: str, age=True, lambda_age: Callable = lambda x: x) -> List[tio.Subject]:
    subjects = []
    df = pd.read_csv(dataset_path)
    for index, row in df.iterrows():
        if age is False:
            subject = tio.Subject(
                image=tio.ScalarImage(row['image']),
                label=tio.LabelMap(row['label'])
            )
        else:
            age_value = lambda_age(row['age']) if lambda_age is not None else row['age']
            subject = tio.Subject(
                image=tio.ScalarImage(row['image']),
                label=tio.LabelMap(row['label']),
                age=age_value
            )
        subjects.append(subject)
    return subjects



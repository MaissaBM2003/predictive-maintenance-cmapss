import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema

op_cols = {f"op_{i}": Column(float, nullable=False) for i in range(1, 4)}
sensor_cols = {f"s_{i}": Column(float, nullable=False) for i in range(1, 22)}

raw_schema = DataFrameSchema(
    {
        "unit": Column(int, Check.ge(1)),
        "cycle": Column(int, Check.ge(1)),
        **op_cols,
        **sensor_cols,
    },
    strict=True,
    coerce=True,
)
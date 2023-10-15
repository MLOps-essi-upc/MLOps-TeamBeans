import great_expectations as gx
import pandas as pd
from great_expectations.datasource import PandasDatasource
import os,sys

script_dir = os.path.dirname(os.path.realpath(__file__))

parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

sys.path.append(parent_dir)

from data import toDF_imgMetadata


train_df, test_df, validation_df = toDF_imgMetadata.main()

# Great Expectations

context = gx.get_context()


context.add_or_update_expectation_suite("beans_suite")
datasource = context.sources.add_or_update_pandas(name="beans_dataset")


for asset_name, asset_df in [("train", train_df), ("test", test_df), ("validation", validation_df)]:

    asset = datasource.add_dataframe_asset(name=asset_name, dataframe=asset_df)
    batch_request = asset.build_batch_request()
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="beans_suite",
        datasource_name="beans_dataset",
        data_asset_name=asset_name,
    )

    column_list = ['image_file_path', 'labels', 'image_size_bytes', 'image_extension',
                   'image_width', 'image_height', 'red_mean', 'green_mean', 'blue_mean']

    validator.expect_table_columns_to_match_ordered_list(column_list=column_list)

    validator.expect_column_values_to_be_in_set("labels", value_set=[0, 1, 2])

    for column_name in column_list:
        validator.expect_column_values_to_not_be_null(column_name)
        if column_name in ('red_mean', 'blue_mean', 'green_mean'):
            validator.expect_column_values_to_be_of_type(column_name, "float", parse_strings_as_datetimes=False)
        elif column_name in ('image_size_bytes', 'image_width', 'image_height'):
            validator.expect_column_values_to_be_of_type(column_name, "int", parse_strings_as_datetimes=False)

    validator.expect_column_values_to_be_in_set(
        column="image_extension",
        value_set=["jpg", "jpeg", "jfif", "pjpeg", "pjp", "png"]
    )

    validator.expect_column_values_to_be_between(column="red_mean", min_value=0, max_value=255, parse_strings_as_datetimes=False)
    validator.expect_column_values_to_be_between(column="green_mean", min_value=0, max_value=255, parse_strings_as_datetimes=False)
    validator.expect_column_values_to_be_between(column="blue_mean", min_value=0, max_value=255, parse_strings_as_datetimes=False)
    
    validator.expect_column_distinct_values_to_equal_set("image_height", {500})
    validator.expect_column_distinct_values_to_equal_set("image_width", {500})
    validator.save_expectation_suite(discard_failed_expectations=False)

    checkpoint = context.add_or_update_checkpoint(
        name=f"checkpoint_{asset_name}",
        validator=validator,
    )




    checkpoint_result = checkpoint.run()
    context.view_validation_result(checkpoint_result)






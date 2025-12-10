# forms.py

from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField, IntegerField
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms.validators import DataRequired,  NumberRange

class DynamicAIForm(FlaskForm):
    submit = SubmitField("Execute")


class InferenceForm(FlaskForm):
    problem_id = SelectField(
        'Problem ID',
        choices=[],
        coerce=str,
        validators=[DataRequired(message='Please select a problem ID')]
    )
    upload_image = FileField(
        'Upload File',
        validators=[
            FileRequired(message='File is required'),
            FileAllowed(['jpg', 'jpeg', 'png', 'gif'], message='Images only!')
        ]
    )
    submit = SubmitField('Predict')

class BulkInferenceForm(FlaskForm):
    problem_id = SelectField(
        'Problem ID',
        choices=[],
        coerce=str,
        validators=[DataRequired(message='Please select a problem ID')]
    )
    input_images_folder_path = StringField(
        'Input Images Folder Path',
        validators=[DataRequired(message='Please enter images folder path')]
    )
    submit = SubmitField('Predict')


class TrainForm(FlaskForm):
    models_list = SelectField(
        'Models List',
        choices=[],
        coerce=str,
        validators=[DataRequired(message='Please select a model')]
    )
    epochs = IntegerField('Epochs', default=50, validators=[DataRequired(), NumberRange(min=20, max=1000)])
    imgsz = IntegerField('Image Size', default=640, validators=[DataRequired(), NumberRange(min=128, max=2048)])
    batch_size = IntegerField('Batch Size', default=16, validators=[DataRequired(), NumberRange(min=1, max=256)])
    dataset_yaml_path = StringField(
        'Dataset Yaml Path',
        validators=[DataRequired(message='Please enter path to the dataset.yaml path')]
    )
    experiment_name = StringField(
        'Experiment Name',
        validators=[DataRequired(message='Please enter experiment name')]
    )
    submit = SubmitField('Train')


class PrepareDatasetForm(FlaskForm):
    models_list = SelectField(
        'Models List',
        choices=[],
        coerce=str,
        validators=[DataRequired(message='Please select a model')]
    )
    input_images_folder_path = StringField(
        'Input Images Folder Path',
        validators=[DataRequired(message='Please enter images folder path')]
    )
    input_annotations_folder_path = StringField(
        'Input Annotations Folder Path',
        validators=[DataRequired(message='Please enter annotations folder path')]
    )
    submit = SubmitField('Prepare Dataset')

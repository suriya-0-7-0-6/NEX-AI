from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileAllowed, FileRequired


class AiInferenceForm(FlaskForm):
    problem_id = SelectField(
        'Problem ID',
        choices=[],
        coerce=str,
        render_kw={'class': 'form-select', 'aria-label': 'Default select example'},
        validators=[DataRequired(message='Please select a problem ID')]
    )

    upload_image = FileField(
        'Upload File',
        validators=[
            FileRequired(message='File is required'),
            FileAllowed(['jpg', 'jpeg', 'png', 'gif'], message='Images only!')
        ]    
    )

    submit = SubmitField(
        'Predict',
    )
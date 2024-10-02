from django.contrib.auth.password_validation import BasePasswordValidator
from django.utils.translation import gettext as _

class NumericPasswordValidator(BasePasswordValidator):
    def validate(self, password, user=None):
        # Allow all numeric passwords
        if password.isdigit():
            return  # Valid password, do nothing
        # Raise an error if the password is not numeric
        raise ValidationError(
            _("This password is not allowed. Numeric passwords are accepted."),
            code='password_no_numeric',
        )

    def get_help_text(self):
        return _("Your password must be a numeric password.")
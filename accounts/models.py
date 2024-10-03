from django.contrib.auth.models import AbstractUser
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


class CustomUser(AbstractUser):
    USER_TYPES = (
        ('VPAA', 'Vice President of Academic Affairs'),
        ('IDEALS', 'Ideals'),
        ('HR', 'HR'),
        ('DEANS', 'Deans'),
    )

    user_type = models.CharField(max_length=20, choices=USER_TYPES)

    def __str__(self):
        return self.username

class UploadCSV (models.Model):
    survey_name = models.CharField(max_length=255)
    date_start = models.CharField(max_length=255)
    date_close = models.CharField(max_length=255)
    crs_num = models.CharField(max_length=255)
    crs_name = models.CharField(max_length=255)
    crs_year = models.IntegerField()
    dept_name = models.CharField(max_length=255)
    crs_dir = models.CharField(max_length=255)
    resp_fac = models.CharField(max_length=255)
    eval_id = models.CharField(max_length=255)
    eval_uname = models.CharField(max_length=255)
    eval_email = models.CharField(max_length=255)
    t_submit = models.CharField(max_length=255)
    mobile = models.CharField(max_length=255)
    grad_year = models.CharField(max_length=255)
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other')
    ]
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default="Not specified")
    program = models.CharField(max_length=255)
    research_1 = models.CharField(max_length=255)
    research_2 = models.CharField(max_length=255)
    research_3 = models.CharField(max_length=255)
    question_1 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor has set clear standards regarding '
                  'their timeliness in responding to messages.')
    question_2 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor has provided the appropriate information and contact details for technical concerns.')
    question_3 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor showed interest in student progress.')
    question_4 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor encourages learners to participate.')
    question_5 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor implements Small Group Discussions (Breakout Rooms).')
    question_6 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor provides equal opportunities for students to share ideas and viewpoints.')
    question_7 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor requires learners to participate.')
    question_8 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor provides platforms for Small Group Discussions.')
    question_9 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='Tasks deployed require collaboration among students.')
    question_10 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor interactively engages learners in a variety of instructional delivery methods.')
    question_11 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor deploys various active learning techniques (role-playing  case studies  group projects  think-pair-share  debates  etc.)')
    question_12 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor provides students access to resources that enrich the course content.')
    question_13 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor demonstrates a thorough knowledge of the subject matter.')
    question_14 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor provides examples  case studies  or problem sets that required higher-order thinking skills.')
    question_15 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor explains the significance of the lessons.')
    question_16 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='Tasks and assessments increase their level of challenge.')
    question_17 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor provides examples or models of what is expected.')
    question_18 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='Course outcomes are clearly presented.')
    question_19 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='Course deadlines are clearly set.')
    question_20 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='Provides reminders regarding upcoming deadlines.')
    question_21 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor provides clear guidelines and/or rubrics of the learning activities.')
    question_22 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor provides Information Feedback (Evaluation of Work  Answers to Questions  Comments on Submissions).')
    question_23 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor provides Acknowledgment Feedback (acknowledges communication sent by student).')
    question_24 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor returns assessments in a timely fashion.')
    question_25 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor engages students in learning activities that address a variety of learning styles and preferences.')
    question_26 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor adapts learning activities to accommodate student&#39s needs.')
    question_27 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='Quality of work is praised.')
    question_28 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor is effective in helping students develop their knowledge  skills  and personalities enabling them to achieve the intended learning outcomes.')
    question_29 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='Course assessments are authentic and measure Higher-Order Thinking Skills.')
    question_30 = models.IntegerField(validators=[
        MinValueValidator(-1),
        MaxValueValidator(5)
    ], default=0,
        help_text='The instructor provides sufficient attempts in the attainment of learning outcomes.')
    question_31 = models.CharField(max_length=255)
    question_32 = models.CharField(max_length=255)

    def __str__(self):
        return (f"Survey: {self.survey_name}, "
                f"Date_start: {self.date_start}, "
                f"Date_close: {self.date_close}, "
                f"Crs_num: {self.crs_num}, "
                f"Crs_name: {self.crs_name}, "
                f"Crs_year: {self.crs_year}, "
                f"Dept_name: {self.dept_name}, "
                f"Crs_dir: {self.crs_dir}, "
                f"Resp_fac: {self.resp_fac}, "
                f"Eval_id: {self.eval_id}, "
                f"Eval_uname: {self.eval_uname}, "
                f"Eval_email: {self.eval_email}, "
                f"T_submit: {self.t_submit}, "
                f"Mobile: {self.mobile}, "
                f"Grad_year: {self.grad_year}, "
                f"Gender: {self.gender},"
                f"Program: {self.program}, "
                f"Research_1: {self.research_1}, "
                f"Research_2: {self.research_2}, "
                f"Research_3: {self.research_3}, "

                f"Question_1: {self.question_1}, "
                f"Question_2: {self.question_2}, "
                f"Question_3: {self.question_3}, "
                f"Question_4: {self.question_4}, "
                f"Question_5: {self.question_5}, "
                f"Question_6: {self.question_6}, "
                f"Question_7: {self.question_7}, "
                f"Question_8: {self.question_8}, "
                f"Question_9: {self.question_9}, "
                f"Question_10: {self.question_10}, "
                f"Question_11: {self.question_11}, "
                f"Question_12: {self.question_12}, "
                f"Question_13: {self.question_13}, "
                f"Question_14: {self.question_14}, "
                f"Question_15: {self.question_15}, "
                f"Question_16: {self.question_16}, "
                f"Question_17: {self.question_17}, "
                f"Question_18: {self.question_18}, "
                f"Question_19: {self.question_19}, "
                f"Question_20: {self.question_20}, "
                f"Question_21: {self.question_21}, "
                f"Question_22: {self.question_22}, "
                f"Question_23: {self.question_23}, "
                f"Question_24: {self.question_24}, "
                f"Question_25: {self.question_25}, "
                f"Question_26: {self.question_26}, "
                f"Question_27: {self.question_27}, "
                f"Question_28: {self.question_28}, "
                f"Question_29: {self.question_29}, "
                f"Question_30: {self.question_30}, "
                f"Question_31: {self.question_31}, "
                f"Question_32: {self.question_32} ")

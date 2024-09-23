from django.contrib import admin
from .models import CustomUser, UploadCSV

class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('username', 'user_type')
    fields = ('username', 'user_type')

class UploadCSVAdmin(admin.ModelAdmin):
    list_display = ('survey_name', 'date_start', 'date_close', 'crs_num', 'crs_name',
                    'crs_year', 'dept_name', 'crs_dir', 'resp_fac', 'eval_id', 'eval_uname',
                    'eval_email', 't_submit', 'mobile', 'grad_year', 'gender', 'program',
                    'research_1', 'research_2', 'research_3',
                    'question_1', 'question_2', 'question_3', 'question_4', 'question_5',
                    'question_6', 'question_7', 'question_8', 'question_9', 'question_10',
                    'question_11', 'question_12', 'question_13', 'question_14', 'question_15',
                    'question_16', 'question_17', 'question_18', 'question_19', 'question_20',
                    'question_21', 'question_22', 'question_23', 'question_24', 'question_25',
                    'question_26', 'question_27', 'question_28', 'question_29', 'question_30',
                    'question_31', 'question_32',
                    )
    search_fields = ('survey_name', 'date_start', 'date_close', 'crs_num', 'crs_name',
                    'crs_year', 'dept_name', 'crs_dir', 'resp_fac', 'eval_id', 'eval_uname',
                    'eval_email', 't_submit', 'mobile', 'grad_year', 'gender', 'program',
                    'research_1', 'research_2', 'research_3',
                    'question_1', 'question_2', 'question_3', 'question_4', 'question_5',
                    'question_6', 'question_7', 'question_8', 'question_9', 'question_10',
                    'question_11', 'question_12', 'question_13', 'question_14', 'question_15',
                    'question_16', 'question_17', 'question_18', 'question_19', 'question_20',
                    'question_21', 'question_22', 'question_23', 'question_24', 'question_25',
                    'question_26', 'question_27', 'question_28', 'question_29', 'question_30',
                    'question_31', 'question_32',
                    )
    list_filter = ('survey_name', 'date_start', 'date_close', 'crs_num', 'crs_name',
                    'crs_year', 'dept_name', 'crs_dir', 'resp_fac', 'eval_id', 'eval_uname',
                    'eval_email', 't_submit', 'mobile', 'grad_year', 'gender', 'program',
                    'research_1', 'research_2', 'research_3',
                    'question_1', 'question_2', 'question_3', 'question_4', 'question_5',
                    'question_6', 'question_7', 'question_8', 'question_9', 'question_10',
                    'question_11', 'question_12', 'question_13', 'question_14', 'question_15',
                    'question_16', 'question_17', 'question_18', 'question_19', 'question_20',
                    'question_21', 'question_22', 'question_23', 'question_24', 'question_25',
                    'question_26', 'question_27', 'question_28', 'question_29', 'question_30',
                    'question_31', 'question_32',
                    )


    # Register your models here.
admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(UploadCSV, UploadCSVAdmin)
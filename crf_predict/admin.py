from django.contrib import admin

# Register your models here.
from crf_predict.models import Doc, Ent
admin.site.register(Doc)
admin.site.register(Ent)

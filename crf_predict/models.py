from django.db import models

# Create your models here.
class Doc(models.Model):
    title = models.CharField(max_length=100)
    text = models.CharField(max_length=10000)
    game = models.CharField(max_length=100)

    def __str__(self):
        return self.title

# Create your models here.
class TestDoc(models.Model):
    title = models.CharField(max_length=100)
    text = models.CharField(max_length=10000)
    game = models.CharField(max_length=100)

    def __str__(self):
        return self.title

class Ent(models.Model):
    doc = models.ForeignKey(Doc, on_delete=models.CASCADE)
    label = models.CharField(max_length=20)
    text = models.CharField(max_length=100)
    start = models.IntegerField()
    end = models.IntegerField()

    def __str__(self):
        return self.text

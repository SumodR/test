# Generated by Django 4.2.1 on 2024-09-05 14:50

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0021_auto_20220914_2104'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='cartitems',
            options={'default_permissions': ('add', 'change', 'delete', 'view'), 'verbose_name': 'Cart Item', 'verbose_name_plural': 'Cart Items'},
        ),
    ]
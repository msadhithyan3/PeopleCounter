from django.conf.urls import url
from detection_app import views

urlpatterns = [
    url(r'^$', views.HomePageView.as_view()),
    url(r'^getPeopleCount$', views.GetPeopleCount.as_view()),  # Add this /getPeopleCount/ route
]

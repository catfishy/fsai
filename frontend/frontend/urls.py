from django.conf.urls import patterns, url
from rest_framework_nested import routers

from authentication.views import AccountViewSet
from frontend.views import IndexView

router = routers.SimpleRouter()
router.register(r'accounts', AccountViewSet)

urlpatterns = patterns(
    '',
    url(r'^api/v1/', include(router.urls)),
    
    url('^.*$', IndexView.as_view(), name='index'),
)

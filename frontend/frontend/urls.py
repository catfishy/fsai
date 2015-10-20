from django.conf.urls import patterns, url, include
from rest_framework_nested import routers

from authentication.views import AccountViewSet, LoginView,  LogoutView
from frontend.views import IndexView
from posts.views import AccountPostsViewSet, PostViewSet
from nba_stats.views import DailyPlayerOwnVectors, DailyPlayerAgainstPosVectors, \
                            DailyPlayerPosTrendVectors, DailyPlayerHomeRoadSplitVectors, \
                            DailyPlayerOppSplitVectors, DailyPlayerMetaVectors, \
                            DailyTeamOwnVectors, DailyTeamOppVectors, DailyTeamSeasonVectors, \
                            DailyTeamMetaVectors



router = routers.SimpleRouter()
router.register(r'accounts', AccountViewSet)
router.register(r'posts', PostViewSet)

accounts_router = routers.NestedSimpleRouter(router, r'accounts', lookup='account')
accounts_router.register(r'posts', AccountPostsViewSet)


urlpatterns = patterns(
    '',
    url(r'^api/v1/', include(router.urls)),
    url(r'^api/v1/', include(accounts_router.urls)),
    url(r'^api/v1/auth/login/$', LoginView.as_view(), name='login'),
    url(r'^api/v1/auth/logout/$', LogoutView.as_view(), name='logout'),
    url(r'^api/v1/nba/daily-team-own', DailyTeamOwnVectors.as_view(), name='daily-team-own'),
    url(r'^api/v1/nba/daily-team-opp', DailyTeamOppVectors.as_view(), name='daily-team-opp'),
    url(r'^api/v1/nba/daily-team-season', DailyTeamSeasonVectors.as_view(), name='daily-team-season'),
    url(r'^api/v1/nba/daily-team-meta', DailyTeamMetaVectors.as_view(), name='daily-team-meta'),
    url(r'^api/v1/nba/daily-player-own', DailyPlayerOwnVectors.as_view(), name='daily-player-own'),
    url(r'^api/v1/nba/daily-player-against', DailyPlayerAgainstPosVectors.as_view(), name='daily-player-against'),
    url(r'^api/v1/nba/daily-player-trend', DailyPlayerPosTrendVectors.as_view(), name='daily-player-trend'),
    url(r'^api/v1/nba/daily-player-homeroadsplit', DailyPlayerHomeRoadSplitVectors.as_view(), name='daily-player-homeroadsplit'),
    url(r'^api/v1/nba/daily-player-oppsplit', DailyPlayerOppSplitVectors.as_view(), name='daily-player-oppsplit'),
    url(r'^api/v1/nba/daily-player-meta', DailyPlayerMetaVectors.as_view(), name='daily-player-meta'),
    url(r'^.*$', IndexView.as_view(), name='index'),
)

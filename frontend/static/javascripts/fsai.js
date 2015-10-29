(function () {
  'use strict';

    angular
        .module('fsai', [
          'ngMaterial',
          'ngAnimate',
          'smart-table',
          'fsai.routes',
          'fsai.authentication',
          'fsai.config',
          'fsai.layout',
          'fsai.posts',
          'fsai.utils',
          'fsai.profiles',
          'fsai.nba'
        ]);

    angular
        .module('fsai.routes', ['ngRoute']);

    angular
        .module('fsai.config', []);

    angular
        .module('fsai')
        .run(run);

    run.$inject = ['$http'];

    /**
    * @name run
    * @desc Update xsrf $http headers to align with Django's defaults
    */
    function run($http) {
      $http.defaults.xsrfHeaderName = 'X-CSRFToken';
      $http.defaults.xsrfCookieName = 'csrftoken';
    }

    /**
    * @name themeConfig
    * @desc Change Theme
    */
    function themeConfig($mdThemingProvider) {
      $mdThemingProvider.theme('default').dark();
    };


})();
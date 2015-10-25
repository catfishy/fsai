(function () {
  'use strict';

  angular
    .module('fsai.authentication', [
      'fsai.authentication.controllers',
      'fsai.authentication.services'
    ]);

  angular
    .module('fsai.authentication.controllers', []);

  angular
    .module('fsai.authentication.services', ['ngCookies']);
})();
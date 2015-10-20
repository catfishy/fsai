(function () {
  'use strict';

  angular
    .module('fsai.posts', [
      'fsai.posts.controllers',
      'fsai.posts.directives',
      'fsai.posts.services'
    ]);

  angular
    .module('fsai.posts.controllers', []);

  angular
    .module('fsai.posts.directives', ['ngDialog']);

  angular
    .module('fsai.posts.services', []);
})();
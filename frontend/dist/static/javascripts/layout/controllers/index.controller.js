!function(){"use strict";function t(t,n,o,s){function e(){function n(t,n,o,s){r.posts=t.data}function e(t,n,o,e){s.error(t.error)}o.all().then(n,e),t.$on("post.created",function(t,n){r.posts.unshift(n)}),t.$on("post.created.error",function(){r.posts.shift()})}var r=this;r.isAuthenticated=n.isAuthenticated(),r.posts=[],e()}angular.module("fsai.layout.controllers").controller("IndexController",t),t.$inject=["$scope","Authentication","Posts","Snackbar"]}();
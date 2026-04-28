// Dean Attali / Beautiful Jekyll 2020
// Refactored to vanilla JS (no jQuery dependency)

var BeautifulJekyllJS = {

  bigImgEl : null,
  numImgs : null,

  init : function() {
    setTimeout(BeautifulJekyllJS.initNavbar, 10);

    // Shorten the navbar after scrolling a little bit down
    window.addEventListener('scroll', function() {
      var navbar = document.querySelector('.navbar');
      if (!navbar) return;
      if (window.scrollY > 50) {
        navbar.classList.add('top-nav-short');
      } else {
        navbar.classList.remove('top-nav-short');
      }
    });

    // On mobile, hide the avatar when expanding the navbar menu
    var mainNavbar = document.getElementById('main-navbar');
    if (mainNavbar) {
      mainNavbar.addEventListener('show.bs.collapse', function () {
        document.querySelector('.navbar').classList.add('top-nav-expanded');
      });
      mainNavbar.addEventListener('hidden.bs.collapse', function () {
        document.querySelector('.navbar').removeClass('top-nav-expanded');
      });
    }

    // show the big header image
    BeautifulJekyllJS.initImgs();

    BeautifulJekyllJS.initSearch();
  },

  initNavbar : function() {
    // Set the navbar-dark/light class based on its background color
    var navbar = document.querySelector('.navbar');
    if (!navbar) return;
    var rgb = getComputedStyle(navbar).backgroundColor.replace(/[^\d,]/g,'').split(",");
    if (rgb.length < 3) return;
    var brightness = Math.round((
      parseInt(rgb[0]) * 299 +
      parseInt(rgb[1]) * 587 +
      parseInt(rgb[2]) * 114
    ) / 1000);
    if (brightness <= 125) {
      navbar.classList.remove('navbar-light');
      navbar.classList.add('navbar-dark');
    } else {
      navbar.classList.remove('navbar-dark');
      navbar.classList.add('navbar-light');
    }
  },

  initImgs : function() {
    // If the page has large images to randomly select from, choose an image
    var bigImgsEl = document.getElementById('header-big-imgs');
    if (bigImgsEl) {
      BeautifulJekyllJS.bigImgEl = bigImgsEl;
      BeautifulJekyllJS.numImgs = bigImgsEl.getAttribute('data-num-img');

      // set an initial image
      var imgInfo = BeautifulJekyllJS.getImgInfo();
      var src = imgInfo.src;
      var desc = imgInfo.desc;
      BeautifulJekyllJS.setImg(src, desc);

      // For better UX, prefetch the next image
      var getNextImg = function() {
        var imgInfo = BeautifulJekyllJS.getImgInfo();
        var src = imgInfo.src;
        var desc = imgInfo.desc;

        var prefetchImg = new Image();
        prefetchImg.src = src;

        setTimeout(function(){
          var img = document.createElement('div');
          img.classList.add('big-img-transition');
          img.style.backgroundImage = 'url(' + src + ')';
          var bigImgEl = document.querySelector('.intro-header.big-img');
          if (bigImgEl) bigImgEl.prepend(img);
          setTimeout(function(){ img.style.opacity = '1'; }, 50);

          setTimeout(function() {
            BeautifulJekyllJS.setImg(src, desc);
            img.remove();
            getNextImg();
          }, 1000);
        }, 6000);
      };

      // If there are multiple images, cycle through them
      if (BeautifulJekyllJS.numImgs > 1) {
        getNextImg();
      }
    }
  },

  getImgInfo : function() {
    var randNum = Math.floor((Math.random() * BeautifulJekyllJS.numImgs) + 1);
    var src = BeautifulJekyllJS.bigImgEl.getAttribute('data-img-src-' + randNum);
    var desc = BeautifulJekyllJS.bigImgEl.getAttribute('data-img-desc-' + randNum);
    return { src : src, desc : desc };
  },

  setImg : function(src, desc) {
    var bigImgEl = document.querySelector('.intro-header.big-img');
    if (bigImgEl) bigImgEl.style.backgroundImage = 'url(' + src + ')';
    var descEl = document.querySelector('.img-desc');
    if (descEl) {
      if (desc !== null && desc !== undefined && desc !== false) {
        descEl.textContent = desc;
        descEl.style.display = '';
      } else {
        descEl.style.display = 'none';
      }
    }
  },

  initSearch : function() {
    if (!document.getElementById('beautifuljekyll-search-overlay')) return;

    var searchLink = document.getElementById('nav-search-link');
    var searchExit = document.getElementById('nav-search-exit');
    var searchInput = document.getElementById('nav-search-input');
    var overlay = document.getElementById('beautifuljekyll-search-overlay');

    if (searchLink) {
      searchLink.addEventListener('click', function(e) {
        e.preventDefault();
        if (overlay) overlay.style.display = 'block';
        if (searchInput) searchInput.focus();
        document.body.classList.add('overflow-hidden');
      });
    }
    if (searchExit) {
      searchExit.addEventListener('click', function(e) {
        e.preventDefault();
        if (overlay) overlay.style.display = 'none';
        document.body.classList.remove('overflow-hidden');
      });
    }
    document.addEventListener('keyup', function(e) {
      if (e.key === 'Escape') {
        if (overlay) overlay.style.display = 'none';
        document.body.classList.remove('overflow-hidden');
      }
    });
  }
};

document.addEventListener('DOMContentLoaded', BeautifulJekyllJS.init);

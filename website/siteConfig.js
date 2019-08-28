/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* List of projects/orgs using your project for the users page */
const authors = [
  {
    caption: "Piotr Bojanowski",
    image: "/img/authors/piotr_bojanowski.jpg",
    infoLink: "https://research.fb.com/people/bojanowski-piotr/",
    pinned: true
  },
  {
    caption: "Edouard Grave",
    image: "/img/authors/edouard_grave.jpeg",
    infoLink: "https://research.fb.com/people/grave-edouard/",
    pinned: true
  },
  {
    caption: "Armand Joulin",
    image: "/img/authors/armand_joulin.jpg",
    infoLink: "https://research.fb.com/people/joulin-armand/",
    pinned: true
  },
  {
    caption: "Tomas Mikolov",
    image: "/img/authors/tomas_mikolov.jpg",
    infoLink: "https://research.fb.com/people/mikolov-tomas/",
    pinned: true
  },
  {
    caption: "Christian Puhrsch",
    image: "/img/authors/christian_puhrsch.png",
    infoLink: "https://research.fb.com/people/puhrsch-christian/",
    pinned: true
  }
];

const users = [
  {
    caption: '"It actually works" - Piotr Bojanowski',
    image: "/img/authors/piotr_bojanowski.jpg",
    infoLink: "https://research.fb.com/people/bojanowski-piotr/",
    pinned: true
  },
  {
    caption: '"It\'s what the kids want" - Santa Claus',
    image: "/img/santa_claus.png",
    infoLink: "https://de.wikipedia.org/wiki/Weihnachtsmann",
    pinned: true
  }
];

const siteConfig = {
  title: "fastText" /* title for your website */,
  url: "https://fasttext.cc" /* your site url */,
  baseUrl: "/" /* base url for your project */,
  repo: "facebookresearch/fastText" /* repo for your project */,
  cname: "fasttext.cc",
  users,
  /* base url for editing docs, usage example: editUrl + 'en/doc1.md' */
  /* editUrl: "https://github.com/facebookresearch/fastText/website/edit/master/docs/", */
  /* header links for links on this site */
  headerLinks: [
    { doc: "support", label: "Docs" },
    { doc: "english-vectors", label: "Resources" },
    { blog: true, label: "Blog" },
    {
      href: "https://github.com/facebookresearch/fastText/",
      label: "GitHub",
      external: true
    }
  ],
  /* path to images for header/footer */
  headerIcon: "img/fasttext-icon-white-web.png",
  disableHeaderTitle: true,
  footerIcon: "img/fasttext-icon-white-web.png",
  favicon: "img/fasttext-icon-bg-web.png",
  mainImg: "img/fasttext-logo-color-web.png",
  /* colors for website */
  colors: {
    primaryColor: "rgb(0,85,129)",
    secondaryColor: "rgb(227,24,35)",
    prismColor:
      "rgba(155,155,155, 0.13)" /* primaryColor in rgba form, with 0.03 alpha */
  },
  separateCss: ["static/docs/en/html"],
  disableTitleTagline: true,
  projectName: "fastText",
  tagline:
    "Library for efficient text classification and representation learning",
  /* remove this to disable google analytics tracking */
  gaTrackingId: "UA-44373548-30",
  ogImage: "img/ogimage.png",
  useEnglishUrl: true,
  scripts: [
    '/tabber.js',
  ],
};

module.exports = siteConfig;

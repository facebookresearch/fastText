/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const siteConfig = {
  title: "fastText" /* title for your website */, //ok
  url: "https://fasttext.cc" /* your site url */, //ok
  baseUrl: "/" /* base url for your project */, //ok
  organizationName: 'facebookresearch',
  projectName: 'fastText'
  favicon: "img/fasttext-icon-bg-web.png", //ok
  mainImg: "img/fasttext-logo-color-web.png",
  /* path to images for header/footer */
  themeConfig: {
    footer: {
      logo: {
        alt: 'Fast Text Logo',
        src: 'img/fasttext-icon-white-web.png'
      }
    },
    image: 'img/ogimage.png',
    navbar: {
      title: 'Fast Text',
      logo: {
        alt: 'Fast Text Logo',
        src: 'img/fasttext-icon-white-web.png'
      },
      /* base url for editing docs, usage example: editUrl + 'en/doc1.md' */
      /* editUrl: "https://github.com/facebookresearch/fastText/website/edit/master/docs/", */
      /* header links for links on this site */
      links: [
        { to: 'support', label: 'Docs' },
        { to: 'english-vectors', label: 'Resources' },
        { to: 'blog', label: 'Blog' },
        {
          href: "https://github.com/facebookresearch/fastText/",
          label: "GitHub",
        }
      ]
    },
    /* remove this to disable google analytics tracking */
    gtag: {
      trackingID: 'UA-44373548-30',
    }
  }
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        theme: {
          customCss: require.resolve('./src/css/custom.css')
        }
      }
    ]
  ]
  
  
  // to change
  separateCss: ["static/docs/en/html"],
  projectName: "fastText", //ok
  tagline:
    "Library for efficient text classification and representation learning", // ok
  scripts: [ //ok
    '/tabber.js',
  ],
};

module.exports = siteConfig;

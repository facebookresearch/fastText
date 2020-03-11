/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

module.exports = {
  title: 'fastText',
  url: 'https://fasttext.cc',
  baseUrl: '/',
  organizationName: 'facebookresearch',
  projectName: 'fastText',
  favicon: 'img/fasttext-icon-bg-web.png',
  tagline: 'Library for efficient text classification and representation learning',
  scripts: [
    '/tabber.js',
  ],
  themeConfig: {
    prism: {
      theme: require('prism-react-renderer/themes/github'),
      darkTheme: require('prism-react-renderer/themes/dracula'),
    },
    footer: {
      logo: {
        alt: 'Fast Text Logo',
        src: 'img/fasttext-icon-white-web.png'
      }
      // add footer links 
    },
    image: 'img/fasttext-logo-color-web.png',
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
        { to: 'docs/support', label: 'Docs' },
        { to: 'docs/english-vectors', label: 'Resources' },
        { to: 'blog', label: 'Blog' },
        {
          href: "https://github.com/facebookresearch/fastText/",
          label: "GitHub",
        }
      ]
    },
    gtag: {
      trackingID: 'UA-44373548-30',
    }
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          path: './docs',
          sidebarPath: require.resolve('./sidebars.json'),
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css')
        }
      }
    ]
  ]
};

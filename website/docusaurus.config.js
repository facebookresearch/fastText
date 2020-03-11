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
  customFields: {
    mainImg: 'img/fasttext-logo-color-web.png'
  },
  themeConfig: {
    prism: {
      theme: require('prism-react-renderer/themes/github'),
      darkTheme: require('prism-react-renderer/themes/dracula'),
    },
    footer: {
      style: 'dark',
      logo: {
        alt: 'Facebook Open Source Logo',
        src: 'https://docusaurus.io/img/oss_logo.png',
        href: 'https://opensource.facebook.com/',
      },
      copyright: `Copyright © ${new Date().getFullYear()} Facebook, Inc.`,
      links: [
        {
          title: 'Support',
          items: [
            {
              label: 'Getting Started',
              to: 'docs/support'
            },
            {
              label: 'Tutorials',
              to: 'docs/supervised-tutorial'
            },
            {
              label: 'FAQs',
              to: 'docs/faqs'
            },
            {
              label: 'API',
              to: 'docs/api'
            }
          ]
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Facebook Group',
              href: 'https://www.facebook.com/groups/1174547215919768/',
            },
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/fasttext',
            },
            {
              label: 'Google Group',
              href: 'https://groups.google.com/forum/#!forum/fasttext-library',
            }
          ]
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              href: 'blog',
            },
            {
              label: 'Github',
              href: 'https://github.com/facebookresearch/fastText'
            },
            {
              label: 'Star',
              href: 'https://github.com/facebookresearch/fastText/'
            }
          ]
        }
      ]
      
    },
    image: 'img/fasttext-logo-color-web.png',
    navbar: {
      title: 'Fast Text',
      logo: {
        alt: 'Fast Text Logo',
        src: 'img/fasttext-icon-white-web.png'
      },
      links: [
        { to: 'docs/support', label: 'Docs', position: 'right' },
        { to: 'docs/english-vectors', label: 'Resources', position: 'right' },
        { to: 'blog', label: 'Blog', position: 'right' },
        {
          href: 'https://github.com/facebookresearch/fastText/',
          label: 'GitHub',
          position: 'right',
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

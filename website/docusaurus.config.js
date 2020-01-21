/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

module.exports = {
  title: 'fastText',
  tagline: 'Library for efficient text classification and representation learning',
  favicon: 'fasttext-icon-bg-web.png',
  url: 'https://fasttext.cc',
  baseUrl: '/',
  organizationName: 'facebook',
  projectName: 'fastText',
  themeConfig: {
    navbar: {
      title: 'FastText',
      logo: {
        alt: 'fastText',
        src: 'img/fasttext-icon-white-web.png',
      },
      links: [
        { to: 'docs/support', label: 'Docs', position: 'right' },
        { to: 'docs/english-vectors', label: 'Resources', position: 'right' },
        { to: 'blog', label: 'Blog', position: 'right' },
        {
          href: 'https://github.com/facebookresearch/fastText/',
          label: 'GitHub',
          position: 'right'
        }
      ],
    },
    footer: {
      style: "dark",
      logo: {
        alt: "fastText",
        src: "img/fasttext-icon-white-web.png"
      },
      links: [
        {
          title: "Support",
          items: [
            {
              label: "Getting Started",
              to: "docs/support"
            },
            {
              label: "Tutorials",
              to: "docs/supervised-tutorial"
            },
            {
              label: "FAQs",
              to: "docs/faqs"
            },
            {
              label: "API",
              to: "docs/api"
            }
          ]
        },
        {
          title: "Community",
          items: [
            {
              label: "Facebook Group",
              href: "https://www.facebook.com/groups/1174547215919768/"
            },
            {
              label: "Stack Overflow",
              href: "https://stackoverflow.com/questions/tagged/fasttext"
            },
            {
              label: "Google Group",
              href: "https://groups.google.com/forum/#!forum/fasttext-library"
            }
          ]
        },
        {
          title: "More",
          items: [
            {
              label: "Blog",
              href: "https://fasttext.cc/blog"
            },
            {
              label: "Github",
              href: "https://github.com/facebookresearch/fastText"
            }
          ]
        }
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Facebook Inc.`
    },
    image: 'img/ogimage.png',
    googleAnalytics: {
      trackingID: 'UA-44373548-30',
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          path: '../docs',
          sidebarPath: require.resolve('./sidebars.json'),
        },
      },
    ],
  ],
  scripts: [
    '/tabber.js',
  ],
};

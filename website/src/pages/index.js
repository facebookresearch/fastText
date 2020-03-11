/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

"use strict";

import React from 'react'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext'
import users from '../data/users'
import useBaseUrl from "@docusaurus/useBaseUrl";
import Link from '@docusaurus/Link';
import Layout from "@theme/Layout";


import './index.css'

const firstContent = [
  {
    content: "FastText is an open-source, free, lightweight library that allows users to learn text representations and text classifiers. It works on standard, generic hardware. Models can later be reduced in size to even fit on mobile devices.",
    title: "What is fastText?",
  }
]

const preTrainedModelsContent = [
  {
    content: "Pre-trained on English webcrawl and Wikipedia",
    image: 'img/model-blue.png',
    imageAlign: "top",
    title: `[English word vectors]('docs/english-vectors')`,
    imageLink: 'docs/english-vectors',
    pinned : "true",
  },
  {
    content: "Pre-trained models for 157 different languages",
    image: 'img/model-red.png',
    imageAlign: "top",
    title: `[Multi-lingual word vectors]("docs/crawl-vectors")`,
    imageLink: 'docs/crawl-vectors',
  },
]

const supportContent = [
  {
    content: "Learn how to use fastText",
    image: "img/fasttext-icon-tutorial.png",
    imageAlign: "top",
    title: `[Tutorials]('docs/supervised-tutorial')`,
    imageLink: "docs/supervised-tutorial",
  },
  {
    content: "Questions gathered from the community",
    image: "img/fasttext-icon-faq.png",
    imageAlign: "top",
    title: `[Frequently Asked Questions]("docs/faqs")`,
    imageLink: "docs/faqs",
  },
  {
    content: "In depth review of fastText commands",
    image: "img/fasttext-icon-api.png",
    imageAlign: "top",
    title: `[API]("docs/api")`,
    imageLink: 'docs/api',
  }
]

const quotes = [
  {
    content: "P. Bojanowski, E. Grave, A. Joulin, T. Mikolov",
    title: "[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)",
  },
  {
    content: "A. Joulin, E. Grave, P. Bojanowski, T. Mikolov",
    title: "[Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)",
  },
  {
    content: "A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jegou, T. Mikolov",
    title: "[FastText.zip: Compressing text classification models](https://arxiv.org/abs/1612.03651)",
  }
]

const Button = ({href, children}) => {
  return (
    <Link to={href}>
        {children}
    </Link>
  )
}

Button.defaultProps = {
  target: "_self"
};

const HomeSplash = () => {
  const context = useDocusaurusContext();
  const { siteConfig = {}} = context;

  return (
    <div className="homeContainer">
      <div className="homeSplashFade">
        <div className="wrapper homeWrapper">
          <div className="inner">
            <img src={`${useBaseUrl(siteConfig.customFields.mainImg)}`} width="50%"/>
            <h2 className="projectTitle">
              <small>{siteConfig.tagline}</small>
            </h2>
            <div className="section promoSection">
              <div className="promoRow">
                <div className="pluginRowBlock">
                  <Button
                    href={useBaseUrl('docs/support')}
                    >
                      Get Started
                    </Button>
                    <Button
                      href={useBaseUrl('docs/english-vectors')}
                    >
                      Download Models
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
    </div>
  )
}

const Index = () => {
  const showcase = users
    .filter(user => {
    return user.pinned;
    })
    .map(user => {
      return (
        <a href={user.infoLink}>
          <img src={user.image} title={user.caption} />
          <br/>
          {user.caption}
        </a>
      );
    });

    return (
      <Layout title="FastText">
        <HomeSplash />
        <div className="mainContainer">
        <div className="container">
          <div class="row">
            {firstContent.map(({ content, title }) => {
            return (
              <div className="col col--4 margin-vert--md">
                <h2>{title}</h2>
                <p>{content}</p>
              </div>
            );
            })}
          </div>
          <h2>Download pre-trained models</h2>
          <div class="row">
            {preTrainedModelsContent.map(({ content, image, imageAlign, title, imageLink, pinned }) => {
            return (
              <div className="col col--4 margin-vert--md">
                <img src={image} />
                <h2>{title}</h2>
                <p>{content}</p>
              </div>
            );
            })}
          </div>
          <h2>Help and references</h2>
          <div class="row">
            {supportContent.map(({ content, image, imageAlign, title, imageLink, pinned }) => {
            return (
              <div className="col col--4 margin-vert--md">
                <img src={image} />
                <h2>{title}</h2>
                <p>{content}</p>
              </div>
            );
            })}
          </div>
          <h2>References</h2>
          <div class="row">
            {quotes.map(({ content, title }) => {
            return (
              <div className="col col--4 margin-vert--md">
                <h2>{title}</h2>
                <p>{content}</p>
              </div>
            );
            })}
          </div>
        </div>
      </div>
    </Layout>
    );
}

export default Index;

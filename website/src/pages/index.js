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
    title: 'English word vectors',
    path: 'docs/english-vectors',
    imageLink: 'docs/english-vectors',
    pinned : "true",
  },
  {
    content: "Pre-trained models for 157 different languages",
    image: 'img/model-red.png',
    imageAlign: "top",
    title: 'Multi-lingual word vectors',
    path: 'docs/crawl-vectors',
    imageLink: 'docs/crawl-vectors',
  },
]

const supportContent = [
  {
    content: "Learn how to use fastText",
    image: "img/fasttext-icon-tutorial.png",
    imageAlign: "top",
    title: 'Tutorials',
    path: 'docs/supervised-tutorial',
    imageLink: "docs/supervised-tutorial",
  },
  {
    content: "Questions gathered from the community",
    image: "img/fasttext-icon-faq.png",
    imageAlign: "top",
    title: 'Frequently Asked Questions',
    path: 'docs/faqs',
    imageLink: "docs/faqs",
  },
  {
    content: "In depth review of fastText commands",
    image: "img/fasttext-icon-api.png",
    imageAlign: "top",
    title: 'API',
    path: "docs/api",
    imageLink: 'docs/api',
  }
]

const quotes = [
  {
    content: "P. Bojanowski, E. Grave, A. Joulin, T. Mikolov",
    title: 'Enriching Word Vectors with Subword Information',
    href: 'https://arxiv.org/abs/1607.04606',
  },
  {
    content: "A. Joulin, E. Grave, P. Bojanowski, T. Mikolov",
    title: 'Bag of Tricks for Efficient Text Classification',
    href: 'https://arxiv.org/abs/1607.01759',
  },
  {
    content: "A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jegou, T. Mikolov",
    title: 'FastText.zip: Compressing text classification models',
    href: 'https://arxiv.org/abs/1612.03651',
  }
]

const Button = ({href, children}) => {
  return (
    <div className="col col--2 margin-horiz--sm">
      <Link
        className="button button--outline button--primary"
        to={href}
      >
        {children}
      </Link>
    </div>
  )
}

Button.defaultProps = {
  target: "_self"
};

const HomeSplash = () => {
  const context = useDocusaurusContext();
  const { siteConfig = {}} = context;

  return (
    <div className="container">
      <img 
        src={`${useBaseUrl(siteConfig.customFields.mainImg)}`}
      />
      <h2 className="hero__title">
        <small className="hero__subtitle">{siteConfig.tagline}</small>
      </h2>
      <div className="row">
            <Button href={useBaseUrl('docs/support')}>
              Get Started
            </Button>
            <Button href={useBaseUrl('docs/english-vectors')}>
              Download Models
            </Button>
        </div>
      </div>
  )
}

const Index = () => {
    return (
      <Layout title="FastText">
        <div className="hero">
        <div className="container">
          <HomeSplash />
          <div className="row">
          {firstContent.map(({ content, title }) => {
            return (
              <div 
                className="descriptionSection"
                id="fast-download"
                style={{textAlign: 'center'}}
              >
              <div className='col'>
                <h1>{title}</h1>
                <p>{content}</p>
              </div>
              </div>
            );
          })}
          </div>
          <Link 
            to={useBaseUrl("docs/english-vectors.htm")}
            style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
            }}
          >
            <h2>Download pre-trained models</h2>
          </Link>
          <div className="row" style={{
            display: 'flex',
            justifyContent: 'center'
          }}>
            {preTrainedModelsContent.map(({ content, image, title, path }) => {
              return (
                <div className="col col--6">
                  <div className="padding-horiz--md" style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      textAlign: 'center'
                     }}>
                    <Link to={path}><img src={image} /></Link>
                    <Link to={path}><h1>{title}</h1></Link>
                    <p>{content}</p>
                  </div>
                </div>
              );
            })}
          </div>
          <Link 
            to={useBaseUrl("docs/support")}
            style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
            }}
          >
            <h2>Help and references</h2>
          </Link>
          <div className="row" style={{
            display: 'flex',
            justifyContent: 'center'
          }}>
            {supportContent.map(({ content, image, title, path }) => {
            return (
              <div className="col col--4">
                <div className="padding-horiz--md" style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  textAlign: 'center'
                }}>
                  <img src={image} />
                  <Link to={path}><h1>{title}</h1></Link>
                  <p>{content}</p>
                </div>
              </div>
            );
            })}
          </div>
          <Link
            to={useBaseUrl("docs/references")}
            style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
            }}
          >
            <h2>References</h2>
          </Link>
          <div className="row" style={{
            display: 'flex',
            justifyContent: 'center'
          }}>
            {quotes.map(({ content, title, href }) => {
            return (
              <div className="col col--4 padding-horiz--m" style={{ display: 'flex', flexDirection: 'column', textAlign: 'center', alignItems: 'center'}}>
                <Link href={href}><h1>{title}</h1></Link>
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

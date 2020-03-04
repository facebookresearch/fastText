/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import classnames from 'classnames';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

const download = [
  {
    image: <img className={styles.itemImage} src="img/model-blue.png" alt="English word vectors" />,
    title: <>English word vectors</>,
    description: (
      <>
        Pre-trained on English webcrawl and Wikipedia
      </>
    ),
    link: "docs/english-vectors"
  },
  {
    image: <img className={styles.itemImage} src="img/model-red.png" alt="Multi-lingual word vectors" />,
    title: <>Multi-lingual word vectors</>,
    description: (
      <>
        Pre-trained models for 157 different languages
      </>
    ),
    link: "docs/crawl-vectors"
  }
];

const help = [
  {
    image: <img className={styles.itemImage} src="img/fasttext-icon-tutorial.png" alt="Tutorials" />,
    title: 'Tutorials',
    description: 'Learn how to use fastText',
    link: 'docs/supervised-tutorial'
  },
  {
    image: <img className={styles.itemImage} src="img/fasttext-icon-faq.png" alt="Frequently asked questions" />,
    title: 'Frequently Asked Questions',
    description: 'Questions gathered from the community',
    link: 'docs/faqs'
  },
  {
    image: <img className={styles.itemImage} src="img/fasttext-icon-api.png" alt="API" />,
    title: 'API',
    description: 'In depth review of fastText commands',
    link: 'docs/api'
  }
];

const references = [
  {
    title: 'Enriching Word Vectors with Subword Information',
    description: 'P. Bojanowski, E. Grave, A. Joulin, T. Mikolov',
    link: 'https://arxiv.org/abs/1607.04606'
  },
  {
    title: 'Bag of Tricks for Efficient Text Classification',
    description: 'A. Joulin, E. Grave, P. Bojanowski, T. Mikolov',
    link: 'https://arxiv.org/abs/1607.01759'
  },
  {
    title: 'FastText.zip: Compressing text classification models',
    description: 'A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jegou, T. Mikolov',
    link: 'https://arxiv.org/abs/1612.03651'
  }
];

function Home() {
  const context = useDocusaurusContext();
  const { siteConfig = {} } = context;

  return (
    <Layout
      title={`${siteConfig.title} - ${siteConfig.tagline}`}
      description={siteConfig.tagline}
    >
      <header className={classnames("hero", styles.heroBanner)}>
        <div className="container">
          <h1 className="hero__title"><img className={styles.heroImage} src={'img/fasttext-logo-color-web.png'} alt="fastText" /></h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className="button button--lg button--outline button--primary"
              to={useBaseUrl('docs/support')}
            >
              Get Started
            </Link>
            &nbsp;
            <Link
              className="button button--lg button--outline button--primary"
              to={useBaseUrl('docs/english-vectors')}
            >
              Download Models
            </Link>
          </div>
        </div>
      </header>

      <main>
        <section className={classnames('hero hero--dark', styles.items)}>
          <div className="container">
            <div className="row">
              <div className="col col--6 col--offset-3 padding-vert--lg">
                <h2>What is fastText?</h2>
                <div className="text--left padding-vert--lg">
                  FastText is an open-source, free, lightweight library that allows users to learn text representations and text classifiers.
                  It works on standard, generic hardware. Models can later be reduced in size to even fit on mobile devices.
                </div>
              </div>
            </div>
          </div>
        </section>

        {download && download.length && (
          <section className={styles.items}>
            <div className="container">
              <h2 className="text--center padding-vert--md">Download pre-trained models</h2>
              <div className="row">
                {download.map(({ image, title, description, link }, idx) => (
                  <div key={idx} className={'col col--6'}>
                    {image && (
                      <div className="text--center padding-vert--md">
                        {image}
                      </div>
                    )}
                    <h3 className="text--center"><Link to={useBaseUrl(link)}>{title}</Link></h3>
                    <p className="text--center">{description}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {help && help.length && (
          <section className={classnames('hero hero--dark', styles.items)}>
            <div className="container">
              <h2 className="text--center padding-vert--md">Help and references</h2>
              <div className="row">
                {help.map(({ image, title, description, link }, idx) => (
                  <div key={idx} className="col col--4">
                    {image && (
                      <div className="text--center padding-vert--md">
                        {image}
                      </div>
                    )}
                    <h3 className="text--center"><Link to={useBaseUrl(link)}>{title}</Link></h3>
                    <p className="text--center">{description}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {references && references.length && (
          <section className={styles.items}>
            <div className="container">
              <h2 className="text--center padding-vert--lg">References</h2>
              <div className="row">
                {references.map(({ title, description, link }, idx) => (
                  <div key={idx} className="col col--4">
                    <h3 className="text--center"><Link to={useBaseUrl(link)}>{title}</Link></h3>
                    <p className="text--center">{description}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </Layout>
  );
}

export default Home;

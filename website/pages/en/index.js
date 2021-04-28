/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require("react");

const CompLibrary = require("../../core/CompLibrary.js");
const Marked = CompLibrary.Marked; /* Used to read markdown */
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

const siteConfig = require(process.cwd() + "/siteConfig.js");

class Button extends React.Component {
  render() {
    return (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={this.props.href} target={this.props.target}>
          {this.props.children}
        </a>
      </div>
    );
  }
}

Button.defaultProps = {
  target: "_self"
};

// head of the page
class HomeSplash extends React.Component {
  render() {
    return (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">
            <div className="inner">
              <img src={siteConfig.baseUrl + siteConfig.mainImg} width="50%"/>
              <h2 className="projectTitle">
                <small>{siteConfig.tagline}</small>
              </h2>
              <div className="section promoSection">
                <div className="promoRow">
                  <div className="pluginRowBlock">
                    <Button
                      href={
                        siteConfig.baseUrl + "docs/" + this.props.language + "/support.html"
                      }
                    >
                      Get Started
                    </Button>
                    <Button
                      href={
                        siteConfig.baseUrl + "docs/" + this.props.language + "/english-vectors.html"
                      }
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
    );
  }
}

class Index extends React.Component {
  render() {
    let language = this.props.language || "en";
    const showcase = siteConfig.users
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
      <div>
        <HomeSplash language={language} />
        <div className="mainContainer">
          <div className="descriptionSection paddingTop lightBackground" style={{ textAlign: "left" }} id="fast-download">
            <Container>
              <GridBlock align="center"
              contents={[
                {
                  content: "FastText is an open-source, free, lightweight library that allows users to learn text representations and text classifiers. It works on standard, generic hardware. Models can later be reduced in size to even fit on mobile devices.",
                  title: "What is fastText?",
                }
              ]}
              layout="twoColumn"
              />
            </Container>
          </div>
          <div
          className="productShowcaseSection paddingTop"
          style={{ textAlign: "center" }} id="fast-download"
          >
          <h2>
              <a href={siteConfig.baseUrl + "docs/en/english-vectors.html"}>Download pre-trained models</a>
          </h2>
          <Container>
            <GridBlock
            align="center"
            contents={[
              {
                content: "Pre-trained on English webcrawl and Wikipedia",
                image: siteConfig.baseUrl + "img/model-blue.png" ,
                imageAlign: "top",
                title: "[English word vectors](" + siteConfig.baseUrl + "docs/en/english-vectors.html)",
                imageLink: siteConfig.baseUrl + "docs/en/english-vectors.html",
                pinned : "true",
              },
              {
                content: "Pre-trained models for 157 different languages",
                image: siteConfig.baseUrl + "img/model-red.png",
                imageAlign: "top",
                title: "[Multi-lingual word vectors](" + siteConfig.baseUrl + "docs/en/crawl-vectors.html)",
                imageLink: siteConfig.baseUrl + "docs/en/crawl-vectors.html",
              },
            ]}
          layout="twoColumn"
            />
            </Container>
          </div>
          <div
          className="productShowcaseSection paddingTop lightBackground"
          style={{ textAlign: "center" }} id="more-info"
          >
          <h2>
              <a href={siteConfig.baseUrl + "docs/en/support.html"}>Help and references</a>
          </h2>
          <Container>
            <GridBlock
              align="center"
              contents={[
                {
                  content: "Learn how to use fastText",
                  image: siteConfig.baseUrl + "img/fasttext-icon-tutorial.png",
                  imageAlign: "top",
                  title: "[Tutorials](" + siteConfig.baseUrl + "docs/en/supervised-tutorial.html)",
                  imageLink: siteConfig.baseUrl + "docs/en/supervised-tutorial.html",
                },
                {
                  content: "Questions gathered from the community",
                  image: siteConfig.baseUrl + "img/fasttext-icon-faq.png",
                  imageAlign: "top",
                  title: "[Frequently Asked Questions](" + siteConfig.baseUrl + "docs/en/faqs.html)",
                  imageLink: siteConfig.baseUrl + "docs/en/faqs.html"
                },
                {
                  content: "In depth review of fastText commands",
                  image: siteConfig.baseUrl + "img/fasttext-icon-api.png",
                  imageAlign: "top",
                  title: "[API](" + siteConfig.baseUrl + "docs/en/api.html)",
                  imageLink: siteConfig.baseUrl + "docs/en/api.html",
                }

              ]}
              layout="threeColumn"
            />
          </Container>
          </div>
          <div className="productShowcaseSection paddingTop">
            <h2>
                <a href={siteConfig.baseUrl + "docs/en/references.html"}>References</a>
            </h2>
            <Container>
              <GridBlock
                align="left"
                contents={[
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

                ]}
                layout="threeColumn"
              />
            </Container>
          </div>
        </div>
      </div>
    );
  }
}
/*
          <div
          className="productShowcaseSection paddingTop"
          style={{ textAlign: "center" }} id="more-info"
          >
          <h2>
            {"Applications"}
          </h2>
          <Container>
            <GridBlock
              align="center"
              contents={[
                {
                  content: "Build a classifier for your usecase.",
                  title: "Classification",
                },
                {
                  content: "Build word vectors on top of your dataset.",
                  title: "Word vectors",
                },
                {
                  content: "Reduce memory footprint and save disk space.",
                  title: "Quantization",
                }

              ]}
              layout="threeColumn"
            />
          </Container>
          <br/>
          <br/>
          </div>
          <div className="productShowcaseSection paddingTop">
            <h2>
              {"Users"}
            </h2>
            <div className="logos indexUsers">
              {showcase}
            </div>
            <br/>
            <br/>
          </div>
 <div className="productShowcaseSection paddingTop lightBackground">
 <h2>
 {"Authors"}
 </h2>
 <div className="logos">
 <a href="https://research.fb.com/people/bojanowski-piotr/">
 <img src="/img/authors/piotr_bojanowski.jpg" title="Piotr Bojanowski" />
 <br />
 Piotr Bojanowski
 </a>
 <a href="https://research.fb.com/people/grave-edouard/">
 <img src="/img/authors/edouard_grave.jpeg" title="Edouard Grave" />
 <br />
 Edouard Grave
 </a>
 <a href="https://research.fb.com/people/joulin-armand/">
 <img src="/img/authors/armand_joulin.jpg" title="Armand Joulin" />
 <br />
 Armand Joulin
 </a>
 <a href="https://research.fb.com/people/mikolov-tomas/">
 <img src="/img/authors/tomas_mikolov.jpg" title="Tomas Mikolov" />
 <br />
 Tomas Mikolov
 </a>
 <a href="https://research.fb.com/people/puhrsch-christian/">
 <img src="/img/authors/christian_puhrsch.png" title="Christian Puhrsch" />
 <br />
 Christian Puhrsch
 </a>
 </div>
 <br/>
 <br/>
 </div>
 */

module.exports = Index;

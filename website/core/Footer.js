/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require("react");

const githubButton = (
  <a
    className="github-button"
    href="https://github.com/facebookresearch/fastText/"
    data-icon="octicon-star"
    data-count-href="/fastText/stargazers"
    data-count-api="/repos/fastText#stargazers_count"
    data-count-aria-label="# stargazers on GitHub"
    aria-label="Star this project on GitHub"
  >
    Star
  </a>
);

class Footer extends React.Component {
  render() {
    const language = this.props.language || "en";
    const currentYear = new Date().getFullYear();
    return (
      <footer className="nav-footer" id="footer">
        <section className="sitemap">
          <a href={this.props.config.baseUrl} className="nav-home">
            <img
              src={this.props.config.baseUrl + this.props.config.footerIcon}
              alt={this.props.config.title}
            />
          </a>
          <div>
            <h5>Support</h5>
            <a
              href={
                this.props.config.baseUrl + "docs/" + language + "/support.html"
              }
            >
              Getting Started
            </a>
            <a
              href={
                this.props.config.baseUrl +
                "docs/" +
                language +
                "/supervised-tutorial.html"
              }
            >
              Tutorials
            </a>
            <a
              href={
                this.props.config.baseUrl +
                "docs/" +
                language +
                "/faqs.html"
              }
            >
              FAQs
            </a>
            <a
              href={
                this.props.config.baseUrl +
                "docs/" +
                language +
                "/api.html"
              }
            >
              API
            </a>
          </div>
          <div>
            <h5>Community</h5>
            <a
              href="https://www.facebook.com/groups/1174547215919768/"
              target="_blank"
            >
              Facebook Group
            </a>
            <a
              href="http://stackoverflow.com/questions/tagged/fasttext"
              target="_blank"
            >
              Stack Overflow
            </a>
            <a
              href="https://groups.google.com/forum/#!forum/fasttext-library"
              target="_blank"
            >
              Google Group
           </a>
          </div>
          <div>
            <h5>More</h5>
            <a href={this.props.config.baseUrl + "blog"}>Blog</a>
            <a href="https://github.com/facebookresearch/fastText" target="_blank">GitHub</a>
            {githubButton}
          </div>
        </section>

        <a
          href="https://code.facebook.com/projects/"
          target="_blank"
          className="fbOpenSource"
        >
          <img
            src={this.props.config.baseUrl + "img/oss_logo.png"}
            alt="Facebook Open Source"
            width="170"
            height="45"
          />
        </a>
        <section className="copyright">
          Copyright &copy; {currentYear} Facebook Inc.
        </section>
      </footer>
    );
  }
}

module.exports = Footer;

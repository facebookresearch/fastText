
[fastText](https://fasttext.cc/) is a library for efficient learning of word representations and sentence classification.

In this document we present how to use fastText in Nodejs with WebAssembly.

```bash
npm install @rafaelkallis/fasttext
```

```js
const FastText = require("@rafaelkallis/fasttext");

main();

async function main() {
	const fastText = await FastText.from("model.bin");
	const [[label, prob]] = fastText.predict("hello world");
	console.log(label, prob);
}
```



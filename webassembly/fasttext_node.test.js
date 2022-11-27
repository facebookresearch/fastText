const fs = require('fs/promises');
const os = require('os');
const fetch = require('node-fetch');
const { FastText } = require('./fasttext_node');

jest.setTimeout(30 * 1000);

describe("FastText WebAssemply", () => {
	let fastText;

	beforeAll(async () => {
		const response = await fetch("https://tickettagger.blob.core.windows.net/models/model-2018-11-12.bin")
		await fs.writeFile(`${os.tmpdir()}/model-test.bin`, await response.buffer());
	});

	beforeEach(async () => {
		fastText = await FastText.from(`${os.tmpdir()}/model-test.bin`);
	});

	const table = [
		["Exception when compiling", "bug"],
		["Idea to improve the product", "enhancement"],
		["I have a question", "question"],
	];
	test.each(table)(".predict('%s')", (text, expectedLabel) => {
		const [[label, prob]] = fastText.predict(text);
		expect(label).toBe(expectedLabel);
		expect(prob).toBeGreaterThan(0.5);
	})
});

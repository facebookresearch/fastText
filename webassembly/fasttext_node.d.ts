declare module "@rafaelkallis/fasttext" {

	/**
	 * A wrapper around the fastText library.
	 */
	export class FastText {
		/**
		 * Creates a new instance of the FastText class.
		 * @param path The path to the fastText model.
		 * @returns A promise that resolves to the FastText model.
		 * @example
		 * const model = await FastText.from("model.bin");
		 */
		public static from(path: string): Promise<FastText>;

		/**
		 * Predicts labels for the given text.
		 * @param text The text to predict.
		 * @returns A promise that resolves to the predicted labels and their probabilities.
		 * @example
		 * const labels = await model.predict("This is a test.");
		 */
		public predict(text: string): Promise<[string, number][]>;
	}
}
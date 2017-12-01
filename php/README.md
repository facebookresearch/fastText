# fastText

[fastText](https://fasttext.cc/) is a library for efficient learning of word representations and sentence classification.

## Installation

```
$ cd php
$ make && sudo make install
```

## Class synopsis

```php
fastText {
    public __construct ( void )
    public int load ( string filename )
    public int getWordRows ( void )
    public int getLabelRows ( void )
    public int getWordId ( string word )
    public string getWord ( int word_id )
    public string getLabel ( int label_id )
    public array getWordVectors ( string word )
    public array getSentenceVectors ( string sentence )
    public mixed getPredict ( streing word [, int k] )
    public mixed getNN ( streing word [, int k] )
    public mixed getAnalogies ( streing word [, int k] )
    public mixed getNgramVectors ( streing word )
    public string lastErrorMsg ( void )
}
```  

## Table of Contents 

[fastText::__construct](#__construct)  
[fastText::load](#load)  
[fastText::getWordRows](#getWordRows)  
[fastText::getLabelRows](#getLabelRows)  
[fastText::getWordId](#getWordId)  
[fastText::getWord](#getWord)  
[fastText::getLabel](#getLabel)  
[fastText::getWordVectors](#getWordVectors)  
[fastText::getSentenceVectors](#getSentenceVectors)  
[fastText::getPredict](#getPredict)  
[fastText::getNN](#getNN)  
[fastText::getAnalogies](#getAnalogies)  
[fastText::getNgramVectors](#getNgramVectors)  
[fastText::lastErrorMsg](#lastErrorMsg)  
  
[return value format](#returnvalf)  

-----

### <a name="__construct">fastText::__construct()

Instantiates a fastText object.  

```php
$ftext = new fastText();
```  

-----

### <a name="load">int fastText::load(string filename)

load a model.  

```php
$model = 'result/model.bin';
$ftext->load($model);
```  

-----

### <a name="getWordRows">int fastText::getWordRows()

get the number of vocabularies.  

```php
$rows = $ftext->getWordRows();
$words = [];
for ($idx = 0; $idx < $rows; $idx++) {
    $words[$idx] = $ftext->getWord($idx);
}
```  

-----

### <a name="getLabelRows">int fastText::getLabelRows()

get the number of labels.  

```php
$rows = $ftext->getLabelRows();
$labels = [];
for ($idx = 0; $idx < $rows; $idx++) {
    $labels[$idx] = $ftext->getLabel($idx);
}
```  

-----

### <a name="getWordId">int fastText::getWordId(string word)

get the word ID within the dictionary.  

```php
$word = 'Bern';
$rowId = $ftext->getWordId($word);
```  

-----

### <a name="getWord">string fastText::getWord(int word_id)

converts a ID into a word.  

```php
$rows = $ftext->getWordRows();
$words = [];
for ($idx = 0; $idx < $rows; $idx++) {
    $words[$idx] = $ftext->getWord($idx);
}
```  

-----

### <a name="getLabel">string fastText::getLabel(int label_id)

converts a ID into a label.  

```php
$rows = $ftext->getLabelRows();
$labels = [];
for ($idx = 0; $idx < $rows; $idx++) {
    $labels[$idx] = $ftext->getLabel($idx);
}
```  

-----

### <a name="getWordVectors">array fastText::getWordVectors(string word)

get the vector representation of word.  

```php
$vectors = $ftext->getWordVectors('Beijing');
print_r($vectors);
```  

-----

### <a name="getSentenceVectors">array fastText::getSentenceVectors(string sentence)

get the vector representation of sentence.  

```php
$sentence = 'It's fine day';

$vectors = $ftext->getSentenceVectors($sentence);
print_r($vectors);
```  

-----

### <a name="getPredict">fastText::getPredict
* array fastText::getPredict(string word)
* FALSE fastText::getPredict(string word)

predict most likely labels with probabilities.  

```php
$probs = $ftext->getPredict('Berlin');
foreach ($probs as $row) {
    echo $row['label'].'  '.$row['prob'];
}
```  

-----

### <a name="getNN">fastText::getNN
* array fastText::getNN(string word)
* FALSE fastText::getNN(string word)

query for nearest neighbors.  

```php
$probs = $ftext->getNN('Washington, D.C.');
foreach ($probs as $row) {
    echo $row['label'].'  '.$row['prob'];
}
```  

-----

### <a name="getAnalogies">fastText::getAnalogies
* array fastText::getAnalogies(string word)
* FALSE fastText::getAnalogies(string word)

query for analogies.  

```php
$probs = $ftext->getAnalogies('Paris + France - Spain');
foreach ($probs as $row) {
    echo $row['label'].'  '.$row['prob'];
}
```  

-----

### <a name="getNgramVectors">fastText::getNgramVectors
* array fastText::getNgramVectors(string word)
* FALSE fastText::getNgramVectors(string word)

get the ngram vectors.  

```php
$res = $ftext->getNgramVectors('London');
print_r($res);
```  

-----

### <a name="lastErrorMsg">string fastText::lastErrorMsg()

get the latest error message.  

```php
$probs = $ftext->getNN('Tokyo');
if (FALSE === $probs) {
    echo $ftext->lastErrorMsg();
}
print_r($probs);
```  

-----

## <a name="returnvalf">return value format

```php
$probs =
[
    ['label'=> '__label__1', 'prob'=> 0.4234 ],
    ['label'=> '__label__2', 'prob'=> 0.2345 ],
    ['label'=> '__label__3', 'prob'=> 0.1456 ],
                        :
                        :
                        :
]
```  

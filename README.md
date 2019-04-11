{::nomarkdown}

<!-- HTML CODE-->
<!DOCTYPE html>
<html>
<head></head>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Self-Driving-Car-Engineer-Nanodegree">Self-Driving Car Engineer Nanodegree<a class="anchor-link" href="#Self-Driving-Car-Engineer-Nanodegree">&#182;</a></h1><h2 id="Deep-Learning">Deep Learning<a class="anchor-link" href="#Deep-Learning">&#182;</a></h2><h2 id="Project:-Build-a-Traffic-Sign-Recognition-Classifier">Project: Build a Traffic Sign Recognition Classifier<a class="anchor-link" href="#Project:-Build-a-Traffic-Sign-Recognition-Classifier">&#182;</a></h2><p>In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary.</p>
<blockquote><p><strong>Note</strong>: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "<strong>File -&gt; Download as -&gt; HTML (.html)</strong>. Include the finished document along with this notebook as your submission.</p>
</blockquote>
<p>In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a <a href="https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md">write up template</a> that can be used to guide the writing process. Completing the code template and writeup template will cover all of the <a href="https://review.udacity.com/#!/rubrics/481/view">rubric points</a> for this project.</p>
<p>The <a href="https://review.udacity.com/#!/rubrics/481/view">rubric</a> contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.</p>
<blockquote><p><strong>Note:</strong> Code and Markdown cells can be executed using the <strong>Shift + Enter</strong> keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.</p>
</blockquote>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Step-0:-Load-The-Data">Step 0: Load The Data<a class="anchor-link" href="#Step-0:-Load-The-Data">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Load pickled data</span>
<span class="kn">import</span> <span class="nn">pickle</span>

<span class="c1"># Completed: Fill this in based on where you saved the training and testing data</span>

<span class="n">training_file</span> <span class="o">=</span> <span class="s1">&#39;./traffic-signs-data/train.p&#39;</span>
<span class="n">validation_file</span><span class="o">=</span><span class="s1">&#39;./traffic-signs-data/valid.p&#39;</span>
<span class="n">testing_file</span> <span class="o">=</span> <span class="s1">&#39;./traffic-signs-data/test.p&#39;</span>


<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">training_file</span><span class="p">,</span> <span class="n">mode</span><span class="o">=</span><span class="s1">&#39;rb&#39;</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
    <span class="n">train</span> <span class="o">=</span> <span class="n">pickle</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">f</span><span class="p">)</span>
<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">validation_file</span><span class="p">,</span> <span class="n">mode</span><span class="o">=</span><span class="s1">&#39;rb&#39;</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
    <span class="n">valid</span> <span class="o">=</span> <span class="n">pickle</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">f</span><span class="p">)</span>
<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">testing_file</span><span class="p">,</span> <span class="n">mode</span><span class="o">=</span><span class="s1">&#39;rb&#39;</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
    <span class="n">test</span> <span class="o">=</span> <span class="n">pickle</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">f</span><span class="p">)</span>
    
<span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;features&#39;</span><span class="p">],</span> <span class="n">train</span><span class="p">[</span><span class="s1">&#39;labels&#39;</span><span class="p">]</span>
<span class="n">X_valid</span><span class="p">,</span> <span class="n">y_valid</span> <span class="o">=</span> <span class="n">valid</span><span class="p">[</span><span class="s1">&#39;features&#39;</span><span class="p">],</span> <span class="n">valid</span><span class="p">[</span><span class="s1">&#39;labels&#39;</span><span class="p">]</span>
<span class="n">X_test</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">test</span><span class="p">[</span><span class="s1">&#39;features&#39;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s1">&#39;labels&#39;</span><span class="p">]</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Step-1:-Dataset-Summary-&amp;-Exploration">Step 1: Dataset Summary &amp; Exploration<a class="anchor-link" href="#Step-1:-Dataset-Summary-&amp;-Exploration">&#182;</a></h2><p>The pickled data is a dictionary with 4 key/value pairs:</p>
<ul>
<li><code>'features'</code> is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).</li>
<li><code>'labels'</code> is a 1D array containing the label/class id of the traffic sign. The file <code>signnames.csv</code> contains id -&gt; name mappings for each id.</li>
<li><code>'sizes'</code> is a list containing tuples, (width, height) representing the original width and height the image.</li>
<li><code>'coords'</code> is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. <strong>THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES</strong></li>
</ul>
<p>Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the <a href="http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html">pandas shape method</a> might be useful for calculating some of the summary results.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Provide-a-Basic-Summary-of-the-Data-Set-Using-Python,-Numpy-and/or-Pandas">Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas<a class="anchor-link" href="#Provide-a-Basic-Summary-of-the-Data-Set-Using-Python,-Numpy-and/or-Pandas">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### Replace each question mark with the appropriate value. </span>
<span class="c1">### Use python, pandas or numpy methods rather than hard coding the results</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="c1"># Completed: Number of training examples</span>
<span class="n">n_train</span> <span class="o">=</span> <span class="n">X_train</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>

<span class="c1"># Completed: Number of validation examples</span>
<span class="n">n_validation</span> <span class="o">=</span> <span class="n">X_valid</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>

<span class="c1"># Completed: Number of testing examples.</span>
<span class="n">n_test</span> <span class="o">=</span> <span class="n">X_test</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>

<span class="c1"># Completed: What&#39;s the shape of an traffic sign image?</span>
<span class="n">image_shape</span> <span class="o">=</span> <span class="n">X_train</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span>

<span class="c1"># Completed: How many unique classes/labels there are in the dataset.</span>
<span class="n">classes</span><span class="p">,</span><span class="n">classes_count</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">unique</span><span class="p">(</span><span class="n">y_train</span><span class="p">,</span><span class="n">return_counts</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">n_classes</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">classes</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Number of training examples =&quot;</span><span class="p">,</span> <span class="n">n_train</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Number of validation examples =&quot;</span><span class="p">,</span> <span class="n">n_validation</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Number of testing examples =&quot;</span><span class="p">,</span> <span class="n">n_test</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Image data shape =&quot;</span><span class="p">,</span> <span class="n">image_shape</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Number of classes =&quot;</span><span class="p">,</span> <span class="n">n_classes</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Include-an-exploratory-visualization-of-the-dataset">Include an exploratory visualization of the dataset<a class="anchor-link" href="#Include-an-exploratory-visualization-of-the-dataset">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.</p>
<p>The <a href="http://matplotlib.org/">Matplotlib</a> <a href="http://matplotlib.org/examples/index.html">examples</a> and <a href="http://matplotlib.org/gallery.html">gallery</a> pages are a great resource for doing visualizations in Python.</p>
<p><strong>NOTE:</strong> It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### Data exploration visualization code goes here.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">matplotlib.image</span> <span class="k">as</span> <span class="nn">mpimg</span>
<span class="kn">import</span> <span class="nn">matplotlib</span> <span class="k">as</span> <span class="nn">mpl</span>

<span class="c1"># Visualizations will be shown in the notebook.</span>
<span class="o">%</span><span class="k">matplotlib</span> inline
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">18</span><span class="p">,</span><span class="mi">4</span><span class="p">))</span>                      <span class="c1"># increase the figure dimensions so it will be more readable</span>

<span class="n">colors</span> <span class="o">=</span> <span class="n">mpl</span><span class="o">.</span><span class="n">cm</span><span class="o">.</span><span class="n">rainbow</span><span class="p">(</span><span class="n">classes</span><span class="o">/</span><span class="p">(</span><span class="n">n_classes</span><span class="o">-</span><span class="mi">1</span><span class="p">))</span>  <span class="c1"># Draw each bar with different color</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;class&#39;</span><span class="p">)</span>    
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Number of images&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">classes</span><span class="p">,</span><span class="n">classes</span><span class="p">)</span>                     <span class="c1"># Draw the class value under each bar</span>

<span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">classes</span><span class="p">,</span><span class="n">classes_count</span><span class="p">,</span><span class="mf">0.7</span><span class="p">,</span><span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">)</span> <span class="c1"># Draw the bar chart</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[3]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;Container object of 43 artists&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
AAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">plotTrafficSignsSummary</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">classes</span><span class="p">,</span> <span class="n">samples_per_class</span><span class="p">,</span> <span class="n">squeeze</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
    <span class="k">for</span> <span class="n">sign</span> <span class="ow">in</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;./signnames.csv&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">values</span><span class="p">:</span>
        <span class="n">class_id</span><span class="o">=</span> <span class="n">sign</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
        <span class="n">class_description</span> <span class="o">=</span> <span class="n">sign</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span>
        <span class="n">class_samples_count</span> <span class="o">=</span> <span class="n">samples_per_class</span><span class="p">[</span><span class="n">class_id</span><span class="p">]</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">{0}</span><span class="s2">. </span><span class="si">{1}</span><span class="s2"> - Samples: </span><span class="si">{2}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">class_id</span><span class="p">,</span> <span class="n">class_description</span><span class="p">,</span><span class="n">class_samples_count</span> <span class="p">))</span>
        <span class="n">sample_indice</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">where</span><span class="p">(</span><span class="n">classes</span><span class="o">==</span><span class="n">class_id</span><span class="p">)[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span>
        <span class="n">fig</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="n">figsize</span> <span class="o">=</span> <span class="p">(</span> <span class="mi">1</span><span class="p">,</span><span class="mi">43</span><span class="p">))</span>
        <span class="n">image</span> <span class="o">=</span> <span class="n">X</span><span class="p">[</span><span class="n">sample_indice</span><span class="p">]</span>
        <span class="n">axis</span> <span class="o">=</span> <span class="n">fig</span><span class="o">.</span><span class="n">add_subplot</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span> <span class="n">xticks</span><span class="o">=</span><span class="p">[],</span> <span class="n">yticks</span><span class="o">=</span><span class="p">[])</span>
        <span class="k">if</span> <span class="n">squeeze</span><span class="p">:</span> <span class="n">image</span> <span class="o">=</span> <span class="n">image</span><span class="o">.</span><span class="n">squeeze</span><span class="p">()</span>
        <span class="k">if</span> <span class="n">cmap</span> <span class="o">==</span> <span class="kc">None</span><span class="p">:</span> <span class="n">axis</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">image</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span> <span class="n">axis</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">image</span><span class="o">.</span><span class="n">squeeze</span><span class="p">(),</span> <span class="n">cmap</span><span class="o">=</span><span class="n">cmap</span><span class="p">)</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span> <span class="p">(</span><span class="s1">&#39;The traffic sign classifier is trained for the following traffic signs&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">()</span>
<span class="n">plotTrafficSignsSummary</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span><span class="n">y_train</span><span class="p">,</span><span class="n">classes_count</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The traffic sign classifier is trained for the following traffic signs

0. Speed limit (20km/h) - Samples: 180
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>1. Speed limit (30km/h) - Samples: 1980
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>2. Speed limit (50km/h) - Samples: 2010
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
AABJRU5ErkJggg==
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>3. Speed limit (60km/h) - Samples: 1260
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
RU5ErkJggg==
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>4. Speed limit (70km/h) - Samples: 1770
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE0AAABNCAYAAADjCemwAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAACzRJREFUeJztXNuO5LgNPZRkV1VfZmY3GWQRIMj/f1Fe8pA8bmZ2Zrcv
dbEkKg88lKt7N0jbwCBYxAQG7qqyZJs6pMhDeqS1hk2WSfhf38DvUTalrZBNaStkU9oK2ZS2Qjal
rZBNaStkU9oK2ZS2QjalrZC05OQYY0tpQEoJwu9UK48KALhOy0TkxfjfTNn8u6tzG2fv45vOY1/N
KfwHAI2/+WcRwX/KEkWkT+Vzny+Xz621j789YpZlSksRf/rhI+4O9zjsdwCAXDIA4On5EYApLwQD
cIp2ThK7TK0TSrXzQ7Rz6jTZZwmYqile0mDHZk81SMM5n21+sXFCI9mnhCb20K60mBLnFDQuZq08
h3OO44DDwe7vcnoAAPzt7//451v0sExpEnC7v0VTwcSHVdjNxN3ebtTu3h6MDxi6FxDkYuer2rE2
IlR1RgXR694jawF4HeFJIdg1StOOVuFiOXKaAPoKaUFciRnHU+GcS7Sw+bRVsghpkIAQRqhGCFcM
MHNzM0hx7P5NmiEmJiKvhe5wqvr4yKkbIog6Lr04uiTYWMzI6aBU7Z8cAY1IbQgd9RBHYeFJBVD7
LoZl2NmQtkIWIa21hqkUDGlAGmyoFlu5IdrnMQ22igBKNhTW7qPQNwl1px+udkpt898AYiAK0eC4
FEeob32qaM2RxTl13kWFSPZNonU/q/16KW5I++ayEGmKPF0Q44hh9JDDVjBI5DF1H6blAgCYiu20
InHeUQmU62Pkb5GoDVe+xmOv/KvdULrbcl+ovg23hiAvd1th+NNqgWp5MfdbZdlG0Bpazaj5DG17
3qCZxFRoGpIBPmz1sKDfVUPw2IuKiX7LWjEMVDw8LLE5gwDC86KbpcyLlWlmE69S/NwW+qqk4ON5
b6p9AcsiJWzmuUqWIQ3mQJtOOJ0sA8jcCHK1YwwAGB4UbgCFaBQAkWY80snTh0NrBujIC+fUaj8G
kY6KkYjppisCVT5Ge2mCwmvYZH6KW4R0N6ALDXRD2gpZjDRbFMV0PgGY0eSoQIo9uHQYhauhY7TV
j1z5zHRMtCJnBrd4mRYVAMHDEaKwiYUzadxhF0bOQcTEkceAUm0zQg+mmb6p9hRuZI78VtmQtkIW
Ik0gEhEkoHK37EFqDyWuwwSbfggMIdAgRGTJhlRw/JD2iMMNAGDcG8sx7OyoANrEQPn8DADdp9Y8
QYLNuU9kVUZDs8aIM8c1sjGezGtrCIl+Fcsy9oW5JyAhQQFE0jeZlEtgCFBK7eHBrDQqsmZMl6P9
pvYQh9tbAMDu8Efc3P8AALi9ZzjDYKABgJu/mrk9/fwJAPDLl88oF6ON3O2HYsoLKSIxZjwXjhd3
/sBu5KIUD1beJpt5rpCFwS1QtSBIgnZSjz81R1pDZFgwMqcLREmeLp27urn7HgCwu38PABhuPyAd
zIGf4LmqO+2Mxu88ZBjf27h3UDz89NnOu9Dp0/lLiwCdfVE3S84TB6SBSGuXRWrYkLZCluWeaFBV
KLQzEZ3Pp/6bClq4DjLQt3upFWBAKcMdACAH82mny4SH048AgMqw4mZvvu3uZsDxaJT006NtAOPO
/NZ+v8PuwzsAwOXLVwBAITWOEjuL4ibhPF5KAeNo93K8LEukNqStkEVIExFITEjp0BmCxl0QndNq
M7vayIddsQ6RSbn7rzOd3MOkwESmlylZO7NY81CQy5HXM6QcGao8N+BuZ6jdH2zc8dFQmS8Twt6u
E3qlyu+l4HKxXbPosjRqmdIgGNKOw+wBB8Y6MfjGUDspWLLdVCMZGVPE4d4ecLyxh7lMZDJaBMBI
vlkMJ2EmKg+3Ni7SLB/O5rwfp4K7aPHdMNo5QUzBpWoPQ5yuCp7sSsbp5GzKVY76BtnMc4UsNs8x
BDSR7vi9wBK7vxUP8tGpBXfGMfWNIBExd7yDIQaUyVb88ReL+j3wDXrGhWHB7c09AGBklhF/fkSE
11kHXs5pcu1mmZjzTs6cBOmbQk9L3ygb0lbIcj5NAEWG853OaOgrRtXPtaMvZeiltEhUpOSpVsFT
NYSVZj5pPHwHAMjHjOejzbF7T3Y3EFWQzv6+dufXnz277CEIpAe+uhBqG9JWyAo+raHW0nsyWvMi
MX1VGBCCF2+5gpMn24rQq7w80hmqZDydv9jfiX6SwWe9SN+Re3LdObvWf+thj7c6yFxkcWa59YC7
dWdW+SxvlYUbgTlQ0dBzTke2m6eGobMcbsKBx1omnJ7N9PbvSCLu7BZyKbgwkj/cfgAAxEQHP1ak
wUz3fKJCad4xVDQ1RWZvkvHKfhy7u8hOZYkvbuzuo9WN5fjmstg8BS87fLxQEhj9a0Ff3cZIWz0v
haAx4M0MToVUtVZFZHbwnm1c/llHweGe4cjJECcXZhYVqGdDViUF7wga0tC9QC3ObrDuKaFX35dW
PjekrZDFvRxaMrSWzhoEp7LFWYSps2GJ4UVivqkiKJP5nQtZC3f2TRuSWDoUmxeNvafjgJTIv92w
wDKZn0zHY6fAG/k0byYUpF64cVpe2eUUQ7zqg1smG9JWyIquoQm1NYC+LA1kEXrvbe5bqno6ww6j
IQ6o9EmPX37scwLA7fs/4M8f/wIAGEeW5HjdONxiTAcAwAf60vOjzXN8vmAqL8Of6sXqApyIsOrt
COTqUsgQP78uC24XbwTagKoNlebYI3pvaWoFtc40ETD3eQxJcNhZPlloNk9fP3NixbQzxTjBuGNL
aqkZgbFUPpqzPz49ATAHr5XhRzJlV8Z+Jz3jwq0g8D57q5aWvij6usf0v8hmnitkOcsxjqiXDNd3
4KomlvRiufRNQUlQsnCOqgk7ZyJogo3lt4ev/5qzCj+Hm4xIQ2NEr69MEbIDgiHUNxWIzVnOz5h4
8RQ8qHVTVKB53XMLOb65rOhPU8QQ5tZ0DzU80BAgeerixYx+rJ0KHxmGjAxkdcq9+u79Gu5+QgCq
93eQFxtY6BWJc8GP7Gy9asn3Xz1nbbSMFgS1Lu1M4/2sGvV/LgtLeAKF8K0U+y4EZxa8mFv7rjm/
YsPVbdqLtmBaExmOhJQgg9cbvBdtfqvFU6N61UkEwAo8wRuOncFwfztgGNipOb0souzC0P1xWIi4
xXXPrIpcSw8xvIHPb7jGgOz9dl50caXV2ju9/TZP/R2AAJDVuLmzsOT7ezs+fvoJz2d76ErT97BG
tSI6ycnmvsbHCnGcqXDPjb0GK/N7NIMsM7jNPFfIiowgo9baCyul0qHvLIRIWiHRckDPHb0NS7X1
VlLPBIpH42JtUwAw8D2Euw9Wknv++hUXbg4VPtecN3ofSajeDOhvvCQIPHxhPhrIriCj7yALsbMh
bYUsDjm0FGhVqNgKntgHkfgaoPkcL++BxxlxM6v762NwFJIXu2UAvI8Rg/e/qW8y3nmM3jvi7fk9
VWoRgY+48wY+IlRa6RuWt5G+VTakrZCFu6f5pdZa364zg8anE9nZmnvvmrO6cytHm1HX29Vn5PQ3
iRkWjGR1xzT2ly7kCmE+p1+g9dZ7JuloncY9sAMJahZxPp2R6Tu1fcOQA7AtWyRgcNbAX0xlv4aq
2iuDmDcAl9ba/PZvfNk/UXPub7acqOxHEo2Q2Gnr/gJsL+w0VOajvaTJ1nHbY+zEUUxpu4PnvHMB
RrCZ5zeX5YWVYM7c87bd4EwBpc25nxOVAdd8Ff8mQmMvwigmDrv7zlpDj5OV+56eH3owW9ni5fms
SOvvp/tLY+7gi85uYGJLqXeGVW29Ez1cdQW8RTakrRBZ8j/1icgnAG/6HwV+p/LXt/wXE4uUtonJ
Zp4rZFPaCtmUtkI2pa2QTWkrZFPaCtmUtkI2pa2QTWkr5N89Zn2oWPrhOQAAAABJRU5ErkJggg==
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>5. Speed limit (80km/h) - Samples: 1650
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>6. End of speed limit (80km/h) - Samples: 360
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
SUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>7. Speed limit (100km/h) - Samples: 1290
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
FLEjX66GAAAAAElFTkSuQmCC
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>8. Speed limit (120km/h) - Samples: 1260
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE0AAABNCAYAAADjCemwAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAACiVJREFUeJztXF2P5bYNPZRs34/52LRpn/s7+tLn/v9/EKBFUGSxSWZ2
7GtbEvtAUrJ9t5uVgUlQwHzRXFuWZfmQPCTlIWbGIXXi/ugJ/D/KsWg75Fi0HXIs2g45Fm2HHIu2
Q45F2yHHou2QY9F2yLFoO6Sp6Xw+X/nh8RlEDsnCL5ImxAAASDEihhkAwCkBABxpJyIkPcYpSpuj
uN8I5/L96AunttfyZuzlMWmdc3nyNuQ8zx+Z+a9fn0jtol2u+Ps//glmD2Z56ClMAIDh1kvbf8Zt
+AwACHrM6/UpJURdrDDLdSGE/DC2HOVZ5QgnBunD+qaVsVgXn8uY5YUk7VPmnqL2ifJCu1MH5/xq
zB9//NcP37IOVYsWY8Tr669w7DHd3gAAN23JiaZPtxvmaZDJ6wRjTPn6pA+YH9TePFFGQX5WhQAz
F7TaKW3XGOO7PqyLe9dzCcPc59vksGk7pAppzAnTdAMnlxE2KarCLMiZ54h5lmOe5A16VYPEWKFH
/8jnDE20aeF9Vk+73lRvMSQMTeS0T+SsAVCVzdcnRtPc28dvkQNpO6QKaSkl9P0N5IBhHgEA0yio
SlERE1O2I+ZR2ZnRBqJ52bS2I0QFfcxrpAkCNyhSpDly4GhjGYqRr7e53PlX5oV7rUPcgbQdUok0
xq0f4FzIrpsNTfq2U4hIeswZh1Na4ZxXflRQZd60YKkIG7cCwdCwpRVxQVW+7FPX6Fujys7VIa1q
0cAJKdwwxzFPjJX/2Ky8I6RgRlcWiBbGnzfuff0IhQRLfxuHwaoUW8JMRGXh7ZhNF4yo/Sm/AM73
svvRoZ7vL3WUA4wUR3AKmZSaWtpbjiGW95ZDFqEcMXFWk+II1NiD4GmtSoY0B4L38n4NxDHx4uo1
UnjR3lEO7RtTKv0qC3IH0nZIpU1jxBg0hlSDbEY/0wXcGWaL+0Aux4zWet9oTwLUNp1PJwDA0Aut
aZoWSZMAzrfaqr1kABoHM62DcoBQzOQaxYl54S7qoHYgbYfUIQ1KDVLxYvmtmu1gxiZCKsIph0On
Ruxca1mLOQFJUDu+SZbEKRrjOGXb5LVtW0FjjIu5KKqYCpWgbaCfPfoCX5UBe7V6phDkbtmgWxvv
upcJF9feKHnrVL3mQWLYMM45hVT6q7oRIUFTUaOoadPJApNzINIUjzsDACIFHSbml0qbmJcWWZXt
wv6WHOq5QyopBzBnilFgDmzzVkunX8STQ6vqdRskQTlPkoxsmhauFaScTlcAgPMlOzJrrJsmaYfp
BgBoPaH1F52DPI5XNDPi2iksWuaS2OTDEby/VCONYRkCoxEbWyE/9JQSV33zHowwCkLmURDz+PQB
AHB5/ABSp9C0nYyZQyWXMyaPakTfXl4AAP3LKxhmTwW9jdISJgfGOnNbwqiCtKapM+0H0nZI1RIT
NJe/OLZCmP7OQbUzWiG3ufU9otqk7ip26/z8JwCAb69oOumXNpUqBoEaQV+Y1Xs+fAcAOKUT5s8/
6RwExV33JL+brgTsZIG7SAIQdwbslTxNshS0KMV9LYFnkwlq7KdpwoMuVnsV491rNcsz4IL0f30V
1TtpZNCdz4hqDj79/AkAcH14lHNNQneSl6Oaj6j3a68dhqkUbgCgMDJCyip7FFbeXaojAoCR0r2L
NqYeY8x5NK9tVAh4AI3GmhYJfPokyGnbDl2ncaWq0u0mkcE89/BaBHkQVoIYXgFI+vvyLOidrLgz
iQpTNy8KKTpRMpykHBOzvyfmX5MDaTuksoRn2Vcg58GM3C4KJZa5aNQR2LmGCBcjro20Xas0oXHw
XmzY+SxweuvFtsUwIQVNj6v9OV3k+peXF1y+F2fivDgZLKrpFhKHcF9gWTqvGjmQtkN22DRg6TGz
eSMrmCRwDlVKYRaQLER3fgAARC9o8p140RgG/PlJaEQ/aM4tCVKb5oRGsyKfe90f0mjohBFOx3JO
t0NAW45winor6KRYNKJkkeuQVr1oKVlxZF2bzGtHy1+byRAQNP3jOj2m2Y40J4A1c5EZvlz/+PQd
eo1VS2Vdq2DokVhpi9qKmDc0uUUt1bjfIvVoiVBXx9MO9dwh1UhjXlfHt9kOLKrhOeorFhdO2b6D
pbsL6Wz12MsgDqAz6sIenUYEXge9koxz/f4CUqRNGi3kTEZgcGsaUWJOQKIMKw7FL+QCvyYH0nZI
dRgFTl+0C5bJoFIM16oHQGq3OARMvWRqL9cnHdHKdAGf/vNvAEBUo3Q6i7H3GDGPSk303m8flY4g
4EHJbQga16rTaM4XDLMQ683WEd07Aj13UI53l0qkSaHCES3yaHbO9maU4glviihjJLwpZcBnCZFO
ln24NHCs73ASzzhMgkoegGelI4OTKY+90IrGNQijwki9b+JFcTrjwioQa8+8foZvkx3qyQAKtrcF
C1Phleg8HXUgjRJub7/KBE4Sb7bnZyQd63qW9qzMvm07gKTf45PwvLYTx5CmHrdXjUNV1e3cnKay
Nzc7gCLmFCrrKod67pFq9XS+UVa9JrDLCrahbrKYU6MAmhjzLCrnoZkIpRVde8KjpTCMrKzGFITO
o8JWk5lj/wsm/btVx+E0mYl5yqhfl1XWFfYj9vwdpNqmhRjUnK0RRouwyt5hMHqg2dlL16LVW876
HYFTxL1OHxFHQVrTaekuj+lykXgaBFUctEATZ5xOSk1Ocv2gJDeElBUip7QX1f8cI9ctwoG0PVJv
05xDSl/4umRJQXIFeU0FQopoNe/faMSeFIUpzXh5GXSwdUE6Ji4faeidGz3XdA9wJxlrzF/B2D4T
dzdPXlg3WtCkGqmPPaFplrSue9pDyVciG+egk5qYEXQBT5aotH0YvssqG1W9KI/jQMbyLSsSdMto
16Kf199iLcPgLY9cfKZVzh2O4P1lRxJSdtvc7ZAomzrgbQ+GJR+N6XPZ6U1OiauSXQJwvQg1cVcr
zGgJMDGi5uEm3dxnObNpHLPDsbn4bPN5Ma/NPBd/H5Tjd5D6TX2Wxdh8VWLlMCK6+xb0fkOp7P+X
39K2BATdGWQV+TGUPWjBto+2co41lqTo8uY/Z3bLcnbO5Qp73sthZJcoh1i1pONA2g7ZsX1UagT/
6/tLQsmt5XyVdVp87GqZj7OiisOcNz33uvcsLZiooYij7aRU6kINfGvI1Awum9de7TqR+S3sWGWc
nmVHutuyHHpgc+cvGVrja54YnS7oWTM1XhOHMcxonFbYtdJ0fpZE5Tz0gFKMECxdLhLiLdOW9ioR
wS+6jWseR8uDZsnfP6w29dXJoZ47ZFdhBVhujhPJDiGlsmNb+5oqnxqPTpU1qioZlXj88IjLw1+k
f/dBW1XdFIEkYwxvN20/yn3HCZOqs3Nyw+dW7t8nj9tk38hvS4682AV+OIJ3F6pZZSL6CcAP7zed
P1z+9i3/YqJq0Q4ROdRzhxyLtkOORdshx6LtkGPRdsixaDvkWLQdcizaDjkWbYf8FwGkj8RLch/H
AAAAAElFTkSuQmCC
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>9. No passing - Samples: 1320
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
2hQyU9oU8t965MfgO9Z86gAAAABJRU5ErkJggg==
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>10. No passing for vehicles over 3.5 metric tons - Samples: 1800
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>11. Right-of-way at the next intersection - Samples: 1170
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
/A/XEYzQB4yThwAAAABJRU5ErkJggg==
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>12. Priority road - Samples: 1890
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>13. Yield - Samples: 1920
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
l1OipgnuAAAAAElFTkSuQmCC
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>14. Stop - Samples: 690
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
kP8CCYnhmXj4TKgAAAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>15. No vehicles - Samples: 540
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>16. Vehicles over 3.5 metric tons prohibited - Samples: 360
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>17. No entry - Samples: 990
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>18. General caution - Samples: 1080
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>19. Dangerous curve to the left - Samples: 180
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
wBZaAn8DYaB2zAXqYd0AAAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>20. Dangerous curve to the right - Samples: 300
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
qbQBmUobkD8An1u2XZl4LaIAAAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>21. Double curve - Samples: 270
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>22. Bumpy road - Samples: 330
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>23. Slippery road - Samples: 450
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
bQL/AjAtev8iGIOmAAAAAElFTkSuQmCC
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>24. Road narrows on the right - Samples: 240
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
QmCC
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>25. Road work - Samples: 1350
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
AElFTkSuQmCC
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>26. Traffic signals - Samples: 540
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
rkJggg==
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>27. Pedestrians - Samples: 210
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
gg==
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>28. Children crossing - Samples: 480
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
wP8AH2Mt/iH/Kc8AAAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>29. Bicycles crossing - Samples: 240
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
AAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>30. Beware of ice/snow - Samples: 390
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
Ky/R2eagAAAAAElFTkSuQmCC
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>31. Wild animals crossing - Samples: 690
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
A0NoHRhC68C/1UtCxoh3IjAAAAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>32. End of all speed and passing limits - Samples: 210
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>33. Turn right ahead - Samples: 599
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
2pm2gf4LCiDDUSDJDCIAAAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>34. Turn left ahead - Samples: 360
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
AAAAAElFTkSuQmCC
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>35. Ahead only - Samples: 1080
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>36. Go straight or right - Samples: 330
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
pg2gkWkD6L9q/0TNW9X8uAAAAABJRU5ErkJggg==
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>37. Go straight or left - Samples: 180
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
5VEAAAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>38. Keep right - Samples: 1860
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>39. Keep left - Samples: 270
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
AAAASUVORK5CYII=
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>40. Roundabout mandatory - Samples: 300
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>41. End of no passing - Samples: 210
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>42. End of no passing by vehicles over 3.5 metric tons - Samples: 210
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
0YsLvVkAAAAASUVORK5CYII=
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Step-2:-Design-and-Test-a-Model-Architecture">Step 2: Design and Test a Model Architecture<a class="anchor-link" href="#Step-2:-Design-and-Test-a-Model-Architecture">&#182;</a></h2><p>Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the <a href="http://benchmark.ini.rub.de/?section=gtsrb&amp;subsection=dataset">German Traffic Sign Dataset</a>.</p>
<p>The LeNet-5 implementation shown in the <a href="https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81">classroom</a> at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!</p>
<p>With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission.</p>
<p>There are various aspects to consider when thinking about this problem:</p>
<ul>
<li>Neural network architecture (is the network over or underfitting?)</li>
<li>Play around preprocessing techniques (normalization, rgb to grayscale, etc)</li>
<li>Number of examples per label (some have more than others).</li>
<li>Generate fake data.</li>
</ul>
<p>Here is an example of a <a href="http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf">published baseline model on this problem</a>. It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Pre-process-the-Data-Set-(normalization,-grayscale,-etc.)">Pre-process the Data Set (normalization, grayscale, etc.)<a class="anchor-link" href="#Pre-process-the-Data-Set-(normalization,-grayscale,-etc.)">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, <code>(pixel - 128)/ 128</code> is a quick way to approximately normalize the data and can be used in this project.</p>
<p>Other pre-processing steps are optional. You can try different techniques to see if it improves performance.</p>
<p>Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include </span>
<span class="c1">### converting to grayscale, etc.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>



<span class="c1"># for CUDA GPU, multiplication is more than 400% faster than division (8 divisions per CPU cycle vs 1.6 division/cycle)</span>
<span class="c1"># http://www.nvidia.com/content/cudazone/download/OpenCL/NVIDIA_OpenCL_ProgrammingGuide.pdf</span>

<span class="c1"># for a 6th generation intel CPU, multiplication takes 1~2 cycles while division takes 24~90 cycles for a 64bit number</span>
<span class="c1"># multiplication is up to 9000% faster</span>
<span class="c1"># http://www.agner.org/optimize/instruction_tables.pdf</span>

<span class="c1"># I modified the normalization equation so it will do the same calculations without division, the devision is pre-calculated</span>
<span class="c1"># (pixel-128)/128 = pixel/128-128/128 = pixel*0.0078125 - 1 </span>

<span class="kn">from</span> <span class="nn">skimage</span> <span class="k">import</span> <span class="n">exposure</span>
<span class="k">def</span> <span class="nf">RGB2NORMALIZED_GREY</span><span class="p">(</span><span class="n">img</span><span class="p">):</span>
    <span class="n">img</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="n">img</span><span class="p">[</span><span class="o">...</span><span class="p">,:</span><span class="mi">3</span><span class="p">],[</span><span class="mf">0.4</span><span class="p">,</span><span class="mf">0.2</span><span class="p">,</span><span class="mf">0.4</span><span class="p">])</span>    <span class="c1"># most traffic signs have red or blue color, no images on the dataset have green color. I assign larger weights to the RED and BLUE color. It also helps to filter out the trees and the grass.  </span>
<span class="c1">#    for i in range(img.shape[0]):</span>
<span class="c1">#        img[i] = exposure.equalize_adapthist(img[i])</span>
    <span class="n">img</span> <span class="o">=</span> <span class="n">img</span><span class="o">*</span><span class="mf">0.0078125</span><span class="o">-</span><span class="mi">1</span>
    <span class="n">img</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">reshape</span><span class="p">(</span><span class="n">img</span><span class="p">,</span><span class="n">img</span><span class="o">.</span><span class="n">shape</span><span class="o">+</span><span class="p">(</span><span class="mi">1</span><span class="p">,))</span>
    <span class="k">return</span> <span class="n">img</span>

<span class="k">def</span> <span class="nf">normalize</span><span class="p">(</span><span class="n">img</span><span class="p">,</span><span class="n">result</span><span class="p">,</span><span class="n">exit_message</span><span class="p">):</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">RGB2NORMALIZED_GREY</span><span class="p">(</span><span class="n">img</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">exit_message</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">result</span>
   
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">cv2</span>

<span class="c1"># convert the initial images to normalized greyscale images</span>
<span class="kn">from</span> <span class="nn">threading</span> <span class="k">import</span> <span class="n">Thread</span>

<span class="c1"># start multiple threads for faster performance</span>

<span class="c1">#Normalized_X_train=[]</span>
<span class="c1">#Normalized_X_valid=[]</span>
<span class="c1">#Normalized_X_test=[]</span>
<span class="c1">#p1 = Thread(target = normalize,args=(X_train,Normalized_X_train,&quot;X_train normalized&quot;))</span>
<span class="c1">#p2 = Thread(target = normalize,args=(X_valid,Normalized_X_valid, &quot;X_valid normalized&quot;))</span>
<span class="c1">#p3 = Thread(target = normalize,args=(X_test,Normalized_X_test, &quot;X_test normalized&quot;))</span>
<span class="c1">#p1.start()</span>
<span class="c1">#p2.start()</span>
<span class="c1">#p3.start()</span>
<span class="c1">#p1.join()</span>
<span class="c1">#p2.join()</span>
<span class="c1">#p3.join()</span>

<span class="n">X_train_grey_normalized</span> <span class="o">=</span> <span class="n">RGB2NORMALIZED_GREY</span><span class="p">(</span><span class="n">X_train</span><span class="p">)</span>

<span class="n">X_valid_grey_normalized</span> <span class="o">=</span> <span class="n">RGB2NORMALIZED_GREY</span><span class="p">(</span><span class="n">X_valid</span><span class="p">)</span>

<span class="n">X_test_grey_normalized</span> <span class="o">=</span> <span class="n">RGB2NORMALIZED_GREY</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># verify that the images monochrome, display an image.</span>
<span class="n">image</span> <span class="o">=</span> <span class="n">X_train_grey_normalized</span><span class="p">[</span><span class="mi">0</span><span class="p">][:,:,</span><span class="mi">0</span><span class="p">]</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">image</span><span class="o">.</span><span class="n">squeeze</span><span class="p">())</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">X_train</span><span class="p">[</span><span class="mi">0</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[8]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;matplotlib.image.AxesImage at 0x24b8653a278&gt;</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
KykHTGQ6IQAAAABJRU5ErkJggg==
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
FMr/A2oQYr/XbnhvAAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># verify that X_train_grey values are between -1 and 1</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;max X_train_grey_normalized value: </span><span class="si">{1}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">amax</span><span class="p">(</span><span class="n">X_train_grey_normalized</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;min X_train_grey_normalized value: </span><span class="si">{1}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">amin</span><span class="p">(</span><span class="n">X_train_grey_normalized</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;max X_valid_grey_normalized value: </span><span class="si">{1}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">amax</span><span class="p">(</span><span class="n">X_valid_grey_normalized</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;min X_valid_grey_normalized value: </span><span class="si">{1}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">amin</span><span class="p">(</span><span class="n">X_valid_grey_normalized</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;max X_test_grey_normalized value: </span><span class="si">{1}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">amax</span><span class="p">(</span><span class="n">X_test_grey_normalized</span><span class="p">))</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;min X_test_grey_normalized value: </span><span class="si">{1}</span><span class="s2">&quot;</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">amin</span><span class="p">(</span><span class="n">X_test_grey_normalized</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>max X_train_grey_normalized value: {1} 0.9921875
min X_train_grey_normalized value: {1} -0.9703125
max X_valid_grey_normalized value: {1} 0.9921875
min X_valid_grey_normalized value: {1} -0.959375
max X_test_grey_normalized value: {1} 0.9921875
min X_test_grey_normalized value: {1} -0.96875
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Model-Architecture">Model Architecture<a class="anchor-link" href="#Model-Architecture">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">rate</span> <span class="o">=</span> <span class="mf">0.005</span>
<span class="n">EPOCHS</span> <span class="o">=</span> <span class="mi">30</span>
<span class="n">BATCH_SIZE</span> <span class="o">=</span> <span class="mi">128</span>  <span class="c1"># The higher the batch size, the faster it runs , if the harware resources are sufficient. However, the quality of the model is reduced, it is more difficult to generalize.</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### Define your architecture here.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>


<span class="k">def</span> <span class="nf">sdim_gr</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>        
    <span class="n">mu</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">sigma</span> <span class="o">=</span> <span class="mf">0.1</span>
    
    <span class="c1"># Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.</span>
    <span class="n">conv1_W</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">6</span><span class="p">),</span> <span class="n">mean</span> <span class="o">=</span> <span class="n">mu</span><span class="p">,</span> <span class="n">stddev</span> <span class="o">=</span> <span class="n">sigma</span><span class="p">))</span>
    <span class="n">conv1_b</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">6</span><span class="p">))</span>
    <span class="n">conv1</span>   <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">conv2d</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">conv1_W</span><span class="p">,</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">&#39;VALID&#39;</span><span class="p">)</span> <span class="o">+</span> <span class="n">conv1_b</span>

    <span class="c1"># Activation.</span>
    <span class="n">conv1</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="n">conv1</span><span class="p">)</span>

    <span class="c1"># Pooling. Input = 28x28x6. Output = 14x14x6.</span>
    <span class="n">conv1</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">max_pool</span><span class="p">(</span><span class="n">conv1</span><span class="p">,</span> <span class="n">ksize</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">&#39;VALID&#39;</span><span class="p">)</span>

    <span class="c1"># Layer 2: Convolutional. Output = 10x10x16.</span>
    <span class="n">conv2_W</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">5</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">6</span><span class="p">,</span> <span class="mi">16</span><span class="p">),</span> <span class="n">mean</span> <span class="o">=</span> <span class="n">mu</span><span class="p">,</span> <span class="n">stddev</span> <span class="o">=</span> <span class="n">sigma</span><span class="p">))</span>
    <span class="n">conv2_b</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">16</span><span class="p">))</span>
    <span class="n">conv2</span>   <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">conv2d</span><span class="p">(</span><span class="n">conv1</span><span class="p">,</span> <span class="n">conv2_W</span><span class="p">,</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">&#39;VALID&#39;</span><span class="p">)</span> <span class="o">+</span> <span class="n">conv2_b</span>
    
    <span class="c1"># Activation.</span>
    <span class="n">conv2</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="n">conv2</span><span class="p">)</span>

    <span class="c1"># Pooling. Input = 10x10x16. Output = 5x5x16.</span>
    <span class="n">conv2</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">max_pool</span><span class="p">(</span><span class="n">conv2</span><span class="p">,</span> <span class="n">ksize</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">&#39;VALID&#39;</span><span class="p">)</span>

    <span class="c1"># Flatten. Input = 5x5x16. Output = 400.</span>
    <span class="n">fc0</span>   <span class="o">=</span> <span class="n">flatten</span><span class="p">(</span><span class="n">conv2</span><span class="p">)</span>
    
    <span class="c1"># Layer 3: Fully Connected. Input = 400. Output = 120.</span>
    <span class="n">fc1_W</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">400</span><span class="p">,</span> <span class="mi">120</span><span class="p">),</span> <span class="n">mean</span> <span class="o">=</span> <span class="n">mu</span><span class="p">,</span> <span class="n">stddev</span> <span class="o">=</span> <span class="n">sigma</span><span class="p">))</span>
    <span class="n">fc1_b</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">120</span><span class="p">))</span>
    <span class="n">fc1</span>   <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">matmul</span><span class="p">(</span><span class="n">fc0</span><span class="p">,</span> <span class="n">fc1_W</span><span class="p">)</span> <span class="o">+</span> <span class="n">fc1_b</span>
    
    <span class="c1"># Activation.</span>
    <span class="n">fc1</span>    <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="n">fc1</span><span class="p">)</span>

    <span class="c1"># Layer 4: Fully Connected. Input = 120. Output = 84.</span>
    <span class="n">fc2_W</span>  <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">120</span><span class="p">,</span> <span class="mi">84</span><span class="p">),</span> <span class="n">mean</span> <span class="o">=</span> <span class="n">mu</span><span class="p">,</span> <span class="n">stddev</span> <span class="o">=</span> <span class="n">sigma</span><span class="p">))</span>
    <span class="n">fc2_b</span>  <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">84</span><span class="p">))</span>
    <span class="n">fc2</span>    <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">matmul</span><span class="p">(</span><span class="n">fc1</span><span class="p">,</span> <span class="n">fc2_W</span><span class="p">)</span> <span class="o">+</span> <span class="n">fc2_b</span>
    
    <span class="c1"># SOLUTION: Activation.</span>
    <span class="n">fc2</span>    <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">relu</span><span class="p">(</span><span class="n">fc2</span><span class="p">)</span>

    <span class="c1"># SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.</span>
    <span class="n">fc3_W</span>  <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">84</span><span class="p">,</span> <span class="mi">43</span><span class="p">),</span> <span class="n">mean</span> <span class="o">=</span> <span class="n">mu</span><span class="p">,</span> <span class="n">stddev</span> <span class="o">=</span> <span class="n">sigma</span><span class="p">))</span>
    <span class="n">fc3_b</span>  <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">43</span><span class="p">))</span>
    <span class="n">logits</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">matmul</span><span class="p">(</span><span class="n">fc2</span><span class="p">,</span> <span class="n">fc3_W</span><span class="p">)</span> <span class="o">+</span> <span class="n">fc3_b</span>
    
    <span class="k">return</span> <span class="n">logits</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Train,-Validate-and-Test-the-Model">Train, Validate and Test the Model<a class="anchor-link" href="#Train,-Validate-and-Test-the-Model">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### Train your model here.</span>
<span class="c1">### Calculate and report the accuracy on the training and validation set.</span>
<span class="c1">### Once a final model architecture is selected, </span>
<span class="c1">### the accuracy on the test set should be calculated and reported as well.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>
<span class="kn">import</span> <span class="nn">tensorflow</span> <span class="k">as</span> <span class="nn">tf</span>
<span class="kn">from</span> <span class="nn">tensorflow.contrib.layers</span> <span class="k">import</span> <span class="n">flatten</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">x</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">placeholder</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">float32</span><span class="p">,</span> <span class="p">(</span><span class="kc">None</span><span class="p">,</span> <span class="mi">32</span><span class="p">,</span> <span class="mi">32</span><span class="p">,</span> <span class="mi">1</span><span class="p">))</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">placeholder</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">int32</span><span class="p">,</span> <span class="p">(</span><span class="kc">None</span><span class="p">))</span>
<span class="n">one_hot_y</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">one_hot</span><span class="p">(</span><span class="n">y</span><span class="p">,</span> <span class="n">n_classes</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">logits</span> <span class="o">=</span> <span class="n">sdim_gr</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
<span class="n">cross_entropy</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">softmax_cross_entropy_with_logits</span><span class="p">(</span><span class="n">labels</span><span class="o">=</span><span class="n">one_hot_y</span><span class="p">,</span> <span class="n">logits</span><span class="o">=</span><span class="n">logits</span><span class="p">)</span>
<span class="n">loss_operation</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">reduce_mean</span><span class="p">(</span><span class="n">cross_entropy</span><span class="p">)</span>
<span class="n">optimizer</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">AdamOptimizer</span><span class="p">(</span><span class="n">learning_rate</span> <span class="o">=</span> <span class="n">rate</span><span class="p">)</span>
<span class="n">training_operation</span> <span class="o">=</span> <span class="n">optimizer</span><span class="o">.</span><span class="n">minimize</span><span class="p">(</span><span class="n">loss_operation</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">correct_prediction</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">equal</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="mi">1</span><span class="p">),</span> <span class="n">tf</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">one_hot_y</span><span class="p">,</span> <span class="mi">1</span><span class="p">))</span>
<span class="n">accuracy_operation</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">reduce_mean</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">cast</span><span class="p">(</span><span class="n">correct_prediction</span><span class="p">,</span> <span class="n">tf</span><span class="o">.</span><span class="n">float32</span><span class="p">))</span>
<span class="n">saver</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">Saver</span><span class="p">()</span>

<span class="k">def</span> <span class="nf">evaluate</span><span class="p">(</span><span class="n">X_data</span><span class="p">,</span> <span class="n">y_data</span><span class="p">):</span>
    <span class="n">num_examples</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">X_data</span><span class="p">)</span>
    <span class="n">total_accuracy</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">sess</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">get_default_session</span><span class="p">()</span>
    <span class="k">for</span> <span class="n">offset</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">num_examples</span><span class="p">,</span> <span class="n">BATCH_SIZE</span><span class="p">):</span>
        <span class="n">batch_x</span><span class="p">,</span> <span class="n">batch_y</span> <span class="o">=</span> <span class="n">X_data</span><span class="p">[</span><span class="n">offset</span><span class="p">:</span><span class="n">offset</span><span class="o">+</span><span class="n">BATCH_SIZE</span><span class="p">],</span> <span class="n">y_data</span><span class="p">[</span><span class="n">offset</span><span class="p">:</span><span class="n">offset</span><span class="o">+</span><span class="n">BATCH_SIZE</span><span class="p">]</span>
        <span class="n">accuracy</span> <span class="o">=</span> <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">accuracy_operation</span><span class="p">,</span> <span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">batch_x</span><span class="p">,</span> <span class="n">y</span><span class="p">:</span> <span class="n">batch_y</span><span class="p">})</span>
        <span class="n">total_accuracy</span> <span class="o">+=</span> <span class="p">(</span><span class="n">accuracy</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">batch_x</span><span class="p">))</span>
    <span class="k">return</span> <span class="n">total_accuracy</span> <span class="o">/</span> <span class="n">num_examples</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.utils</span> <span class="k">import</span> <span class="n">shuffle</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">ConfigProto</span><span class="p">()</span>               <span class="c1"># configure tensorflow session</span>
<span class="n">config</span><span class="o">.</span><span class="n">log_device_placement</span><span class="o">=</span><span class="kc">True</span>        <span class="c1"># log CPU or GPU is used</span>
<span class="n">config</span><span class="o">.</span><span class="n">gpu_options</span><span class="o">.</span><span class="n">allow_growth</span> <span class="o">=</span> <span class="kc">True</span>  <span class="c1"># allow dynamically allocate memory to prevent an error on my system, may not be needed on other systems</span>
<span class="n">sess</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Session</span><span class="p">(</span><span class="n">config</span><span class="o">=</span><span class="n">config</span><span class="p">)</span>

<span class="n">BEST_VALIDATION_ACCURACY</span> <span class="o">=</span> <span class="mi">0</span>            <span class="c1"># store the best train results </span>
<span class="n">valid_accuracy</span><span class="o">=</span><span class="p">[]</span>                  <span class="c1"># store the validation accuracy per epoch</span>
<span class="c1"># test_accuracy=[]                        # store the test accuracy per epoch</span>
<span class="k">with</span> <span class="n">sess</span><span class="p">:</span>   
    <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">global_variables_initializer</span><span class="p">())</span>
    <span class="n">num_examples</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">X_train_grey_normalized</span><span class="p">)</span>
    
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Training...&quot;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">EPOCHS</span><span class="p">):</span>
        <span class="n">X_train_epoch</span><span class="p">,</span> <span class="n">y_train_epoch</span> <span class="o">=</span> <span class="n">shuffle</span><span class="p">(</span><span class="n">X_train_grey_normalized</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">offset</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">num_examples</span><span class="p">,</span> <span class="n">BATCH_SIZE</span><span class="p">):</span>
            <span class="n">end</span> <span class="o">=</span> <span class="n">offset</span> <span class="o">+</span> <span class="n">BATCH_SIZE</span>
            <span class="n">batch_x</span><span class="p">,</span> <span class="n">batch_y</span> <span class="o">=</span> <span class="n">X_train_epoch</span><span class="p">[</span><span class="n">offset</span><span class="p">:</span><span class="n">end</span><span class="p">],</span> <span class="n">y_train_epoch</span><span class="p">[</span><span class="n">offset</span><span class="p">:</span><span class="n">end</span><span class="p">]</span>
            <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">training_operation</span><span class="p">,</span> <span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">batch_x</span><span class="p">,</span> <span class="n">y</span><span class="p">:</span> <span class="n">batch_y</span><span class="p">})</span>
            
        <span class="n">validation_accuracy</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span><span class="n">X_valid_grey_normalized</span><span class="p">,</span> <span class="n">y_valid</span><span class="p">)</span>
        <span class="c1"># test_accuracy.append(evaluate(X_test_grey_normalized,y_test)) t</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;EPOCH </span><span class="si">{}</span><span class="s2"> ...&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">))</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Validation Accuracy = </span><span class="si">{:.3f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">validation_accuracy</span><span class="p">))</span>
        <span class="n">valid_accuracy</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">validation_accuracy</span><span class="p">)</span>
        <span class="c1">#test_accuracy.append(evaluate(X_test_grey_normalized,y_test))</span>
        <span class="c1"># In case of overtraining, I want to keep the best training results, not the last one</span>
        <span class="k">if</span> <span class="n">validation_accuracy</span><span class="o">&gt;</span><span class="n">BEST_VALIDATION_ACCURACY</span><span class="p">:</span>    <span class="c1"># if the results are better than the prevous EPOCH, save the results</span>
            <span class="c1"># it is expected to have validation accuracy better than 93% in this project, </span>
            <span class="c1"># so I ignore all values smaller than 0.93 to save time.</span>
            <span class="c1"># I will check if BEST_VALIDATION_ACCURACY &gt; 0 before I load the results to avoid an error</span>
            <span class="k">if</span> <span class="n">validation_accuracy</span><span class="o">&gt;</span><span class="mf">0.93</span><span class="p">:</span>                    
                <span class="n">BEST_VALIDATION_ACCURACY</span><span class="o">=</span><span class="n">validation_accuracy</span>
                <span class="n">saver</span><span class="o">.</span><span class="n">save</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="s1">&#39;./TrafficSignClassifier&#39;</span><span class="p">)</span>
                <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Model saved&quot;</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Training...

EPOCH 1 ...
Validation Accuracy = 0.830

EPOCH 2 ...
Validation Accuracy = 0.882

EPOCH 3 ...
Validation Accuracy = 0.911

EPOCH 4 ...
Validation Accuracy = 0.910

EPOCH 5 ...
Validation Accuracy = 0.917

EPOCH 6 ...
Validation Accuracy = 0.927

EPOCH 7 ...
Validation Accuracy = 0.912

EPOCH 8 ...
Validation Accuracy = 0.905

EPOCH 9 ...
Validation Accuracy = 0.908

EPOCH 10 ...
Validation Accuracy = 0.927

EPOCH 11 ...
Validation Accuracy = 0.913

EPOCH 12 ...
Validation Accuracy = 0.927

EPOCH 13 ...
Validation Accuracy = 0.916

EPOCH 14 ...
Validation Accuracy = 0.930
Model saved

EPOCH 15 ...
Validation Accuracy = 0.924

EPOCH 16 ...
Validation Accuracy = 0.936
Model saved

EPOCH 17 ...
Validation Accuracy = 0.921

EPOCH 18 ...
Validation Accuracy = 0.927

EPOCH 19 ...
Validation Accuracy = 0.927

EPOCH 20 ...
Validation Accuracy = 0.934

EPOCH 21 ...
Validation Accuracy = 0.938
Model saved

EPOCH 22 ...
Validation Accuracy = 0.933

EPOCH 23 ...
Validation Accuracy = 0.931

EPOCH 24 ...
Validation Accuracy = 0.918

EPOCH 25 ...
Validation Accuracy = 0.943
Model saved

EPOCH 26 ...
Validation Accuracy = 0.944
Model saved

EPOCH 27 ...
Validation Accuracy = 0.930

EPOCH 28 ...
Validation Accuracy = 0.941

EPOCH 29 ...
Validation Accuracy = 0.933

EPOCH 30 ...
Validation Accuracy = 0.939

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Validation accuracy: </span><span class="si">{:.3f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">BEST_VALIDATION_ACCURACY</span><span class="p">))</span>
<span class="c1">## Run the predictions here and use the model to output the prediction for each image.</span>
<span class="c1">### Make sure to pre-process the images with the same pre-processing pipeline used earlier.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>
<span class="n">config</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">ConfigProto</span><span class="p">()</span>               <span class="c1"># configure tensorflow session</span>
<span class="n">config</span><span class="o">.</span><span class="n">log_device_placement</span><span class="o">=</span><span class="kc">True</span>        <span class="c1"># log CPU or GPU is used</span>
<span class="n">config</span><span class="o">.</span><span class="n">gpu_options</span><span class="o">.</span><span class="n">allow_growth</span> <span class="o">=</span> <span class="kc">True</span>  <span class="c1"># allow dynamically allocate memory to prevent an error on my system, may not be needed on other systems</span>
<span class="n">sess</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Session</span><span class="p">(</span><span class="n">config</span><span class="o">=</span><span class="n">config</span><span class="p">)</span>

<span class="k">with</span> <span class="n">sess</span><span class="p">:</span> 
    <span class="n">saver</span><span class="o">.</span><span class="n">restore</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">latest_checkpoint</span><span class="p">(</span><span class="s1">&#39;.&#39;</span><span class="p">))</span>  
    <span class="n">train_accuracy</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span><span class="n">X_train_grey_normalized</span><span class="p">,</span><span class="n">y_train</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Training accuracy: </span><span class="si">{:.3f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">train_accuracy</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Validation accuracy: 0.944
INFO:tensorflow:Restoring parameters from .\TrafficSignClassifier
Training accuracy: 0.994
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">5</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">&quot;EPOCHS&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;Accuracy&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">EPOCHS</span><span class="p">),</span><span class="n">valid_accuracy</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[18]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>[&lt;matplotlib.lines.Line2D at 0x24b9f978cc0&gt;]</pre>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">with</span> <span class="n">tf</span><span class="o">.</span><span class="n">Session</span><span class="p">(</span><span class="n">config</span><span class="o">=</span><span class="n">config</span><span class="p">)</span> <span class="k">as</span> <span class="n">sess</span><span class="p">:</span>
    <span class="n">saver</span><span class="o">.</span><span class="n">restore</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">latest_checkpoint</span><span class="p">(</span><span class="s1">&#39;.&#39;</span><span class="p">))</span>  
    <span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Test Accuracy = </span><span class="si">{:.3f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">evaluate</span><span class="p">(</span><span class="n">X_test_grey_normalized</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>INFO:tensorflow:Restoring parameters from .\TrafficSignClassifier
Test Accuracy = 0.912
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Step-3:-Test-a-Model-on-New-Images">Step 3: Test a Model on New Images<a class="anchor-link" href="#Step-3:-Test-a-Model-on-New-Images">&#182;</a></h2><p>To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.</p>
<p>You may find <code>signnames.csv</code> useful as it contains mappings from the class id (integer) to the actual sign name.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Load-and-Output-the-Images">Load and Output the Images<a class="anchor-link" href="#Load-and-Output-the-Images">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### Load the images and plot them here.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>
<span class="kn">from</span> <span class="nn">scipy.misc</span> <span class="k">import</span> <span class="n">imread</span>
<span class="n">images_directory</span><span class="o">=</span><span class="s1">&#39;./german-signs-images/&#39;</span>
<span class="k">def</span> <span class="nf">cropimread</span><span class="p">(</span><span class="n">fn</span><span class="p">):</span>
    <span class="s2">&quot;Function to crop center the image files to 32x32&quot;</span>
    <span class="n">img_pre</span><span class="o">=</span> <span class="n">imread</span><span class="p">(</span><span class="n">fn</span><span class="p">)</span>
    <span class="n">img</span><span class="o">=</span> <span class="n">img_pre</span><span class="p">[</span><span class="mi">6</span><span class="p">:</span><span class="mi">38</span><span class="p">,</span><span class="mi">6</span><span class="p">:</span><span class="mi">38</span><span class="p">]</span>
    <span class="k">return</span> <span class="n">img</span>
<span class="n">test_images</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">uint8</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">((</span><span class="mi">5</span><span class="p">,</span><span class="mi">32</span><span class="p">,</span><span class="mi">32</span><span class="p">,</span><span class="mi">3</span><span class="p">)))</span>
<span class="c1">#test_images=np.array([cropimread(images_directory + &#39;00000_00002.ppm&#39; ),cropimread(images_directory + &#39;00000_00003.ppm&#39; ),cropimread(images_directory + &#39;00000_00005.ppm&#39; ),cropimread(images_directory + &#39;00007_00002.ppm&#39; ),cropimread(images_directory + &#39;00008_00005.ppm&#39; )])</span>
<span class="n">test_images</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">=</span><span class="n">cropimread</span><span class="p">(</span><span class="n">images_directory</span> <span class="o">+</span> <span class="s1">&#39;1.ppm&#39;</span><span class="p">)</span>
<span class="n">test_images</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">=</span><span class="n">cropimread</span><span class="p">(</span><span class="n">images_directory</span> <span class="o">+</span> <span class="s1">&#39;2.ppm&#39;</span><span class="p">)</span>
<span class="n">test_images</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span><span class="o">=</span><span class="n">cropimread</span><span class="p">(</span><span class="n">images_directory</span> <span class="o">+</span> <span class="s1">&#39;3.ppm&#39;</span><span class="p">)</span>
<span class="n">test_images</span><span class="p">[</span><span class="mi">3</span><span class="p">]</span><span class="o">=</span><span class="n">cropimread</span><span class="p">(</span><span class="n">images_directory</span> <span class="o">+</span> <span class="s1">&#39;4.ppm&#39;</span><span class="p">)</span>
<span class="n">test_images</span><span class="p">[</span><span class="mi">4</span><span class="p">]</span><span class="o">=</span><span class="n">cropimread</span><span class="p">(</span><span class="n">images_directory</span> <span class="o">+</span> <span class="s1">&#39;5.ppm&#39;</span><span class="p">)</span>

<span class="n">test_classes</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span> <span class="p">([</span><span class="mi">5</span><span class="p">,</span><span class="mi">18</span><span class="p">,</span><span class="mi">19</span><span class="p">,</span><span class="mi">28</span><span class="p">,</span><span class="mi">24</span><span class="p">])</span>

<span class="n">class_names</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;./signnames.csv&#39;</span><span class="p">)</span><span class="o">.</span><span class="n">values</span>

<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">5</span><span class="p">):</span>
    <span class="n">fig</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">i</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="nb">format</span><span class="p">(</span><span class="n">class_names</span><span class="p">[</span><span class="n">test_classes</span><span class="p">[</span><span class="n">i</span><span class="p">]][</span><span class="mi">1</span><span class="p">]))</span>
    <span class="n">axis</span> <span class="o">=</span> <span class="n">fig</span><span class="o">.</span><span class="n">add_subplot</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span> <span class="n">xticks</span><span class="o">=</span><span class="p">[],</span> <span class="n">yticks</span><span class="o">=</span><span class="p">[])</span>
    <span class="n">axis</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">test_images</span><span class="p">[</span><span class="n">i</span><span class="p">])</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Speed limit (80km/h)
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
xCH/DZkCKWMbvksVAAAAAElFTkSuQmCC
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>General caution
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Dangerous curve to the left
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Children crossing
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Road narrows on the right
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Predict-the-Sign-Type-for-Each-Image">Predict the Sign Type for Each Image<a class="anchor-link" href="#Predict-the-Sign-Type-for-Each-Image">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### Run the predictions here and use the model to output the prediction for each image.</span>
<span class="c1">### Make sure to pre-process the images with the same pre-processing pipeline used earlier.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>
<span class="n">config</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">ConfigProto</span><span class="p">()</span>               <span class="c1"># configure tensorflow session</span>
<span class="n">config</span><span class="o">.</span><span class="n">log_device_placement</span><span class="o">=</span><span class="kc">True</span>        <span class="c1"># log CPU or GPU is used</span>
<span class="n">config</span><span class="o">.</span><span class="n">gpu_options</span><span class="o">.</span><span class="n">allow_growth</span> <span class="o">=</span> <span class="kc">True</span>  <span class="c1"># allow dynamically allocate memory to prevent an error on my system, may not be needed on other systems</span>
<span class="n">sess</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Session</span><span class="p">(</span><span class="n">config</span><span class="o">=</span><span class="n">config</span><span class="p">)</span>
<span class="n">test_images_grey_normalized</span> <span class="o">=</span> <span class="n">RGB2NORMALIZED_GREY</span><span class="p">(</span><span class="n">test_images</span><span class="p">)</span>
<span class="k">with</span> <span class="n">sess</span><span class="p">:</span> 
    <span class="n">saver</span><span class="o">.</span><span class="n">restore</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">latest_checkpoint</span><span class="p">(</span><span class="s1">&#39;.&#39;</span><span class="p">))</span>  
    <span class="n">test_accuracy</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span><span class="n">test_images_grey_normalized</span><span class="p">,</span><span class="n">test_classes</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">test_accuracy</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>INFO:tensorflow:Restoring parameters from .\TrafficSignClassifier
0.40000000596
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">prob</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">placeholder</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span>
<span class="n">no_image_found</span> <span class="o">=</span> <span class="n">imread</span><span class="p">(</span><span class="s2">&quot;./no_mach_found.bmp&quot;</span><span class="p">)</span>


<span class="k">with</span> <span class="n">tf</span><span class="o">.</span><span class="n">Session</span><span class="p">()</span> <span class="k">as</span> <span class="n">sess</span><span class="p">:</span>
    <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">global_variables_initializer</span><span class="p">())</span>
    <span class="n">saver</span><span class="o">.</span><span class="n">restore</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">latest_checkpoint</span><span class="p">(</span><span class="s1">&#39;.&#39;</span><span class="p">))</span>  
    <span class="n">softmax_logits</span> <span class="o">=</span> <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">softmax</span><span class="p">(</span><span class="n">logits</span><span class="p">),</span> <span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">test_images_grey_normalized</span><span class="p">,</span> <span class="n">prob</span><span class="p">:</span> <span class="mf">1.0</span><span class="p">})</span>
    <span class="n">top_k</span> <span class="o">=</span> <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">top_k</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">softmax</span><span class="p">(</span><span class="n">logits</span><span class="p">),</span><span class="n">k</span><span class="o">=</span><span class="mi">5</span><span class="p">),</span> <span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">test_images_grey_normalized</span><span class="p">,</span> <span class="n">prob</span><span class="p">:</span> <span class="mf">1.0</span><span class="p">})</span>
    
    
    <span class="k">def</span> <span class="nf">display_info</span><span class="p">(</span><span class="n">i</span><span class="p">,</span><span class="n">j</span><span class="p">,</span> <span class="n">index</span><span class="p">,</span><span class="n">probability</span><span class="p">,</span><span class="n">k</span><span class="p">):</span>
         <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;Guess </span><span class="si">{}</span><span class="s1">: </span><span class="si">{}</span><span class="s1"> (</span><span class="si">{:.0f}</span><span class="s1">%)&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">((</span><span class="n">j</span><span class="o">-</span><span class="mi">1</span><span class="o">-</span><span class="mi">6</span><span class="o">*</span><span class="n">i</span><span class="p">)</span><span class="o">+</span><span class="mi">1</span><span class="p">,</span><span class="n">class_names</span><span class="p">[</span><span class="n">probability</span><span class="p">][</span><span class="mi">1</span><span class="p">],</span> <span class="mi">100</span><span class="o">*</span><span class="n">top_k</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="n">i</span><span class="p">][</span><span class="n">k</span><span class="p">]))</span>

    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">5</span><span class="p">):</span>                       <span class="c1"># for each sign</span>
        <span class="n">fig</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">i</span><span class="p">)</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">test_images</span><span class="p">[</span><span class="n">i</span><span class="p">])</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
        <span class="nb">print</span><span class="p">(</span><span class="nb">format</span><span class="p">(</span><span class="n">class_names</span><span class="p">[</span><span class="n">test_classes</span><span class="p">[</span><span class="n">i</span><span class="p">]][</span><span class="mi">1</span><span class="p">]))</span>
        
        <span class="nb">print</span><span class="p">()</span>
        <span class="k">for</span> <span class="n">k</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">5</span><span class="p">):</span>                   <span class="c1"># for each top 5 probability</span>
            <span class="n">probability</span> <span class="o">=</span> <span class="n">top_k</span><span class="p">[</span><span class="mi">1</span><span class="p">][</span><span class="n">i</span><span class="p">][</span><span class="n">k</span><span class="p">]</span>           
            <span class="n">display_info</span><span class="p">(</span><span class="n">i</span><span class="p">,</span><span class="mi">6</span><span class="o">*</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="o">+</span><span class="n">k</span><span class="p">,</span><span class="n">np</span><span class="o">.</span><span class="n">argwhere</span><span class="p">(</span><span class="n">test_classes</span> <span class="o">==</span> <span class="n">probability</span><span class="p">),</span><span class="n">probability</span><span class="p">,</span><span class="n">k</span><span class="p">)</span>
        
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>INFO:tensorflow:Restoring parameters from .\TrafficSignClassifier
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
X4hE+X8UKza3T7mFmAAAAABJRU5ErkJggg==
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Speed limit (80km/h)

Guess 1: Speed limit (30km/h) (51%)
Guess 2: Speed limit (70km/h) (42%)
Guess 3: Speed limit (20km/h) (7%)
Guess 4: Speed limit (120km/h) (0%)
Guess 5: Speed limit (80km/h) (0%)
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>General caution

Guess 1: General caution (100%)
Guess 2: Priority road (0%)
Guess 3: Pedestrians (0%)
Guess 4: Traffic signals (0%)
Guess 5: Right-of-way at the next intersection (0%)
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Dangerous curve to the left

Guess 1: Dangerous curve to the right (100%)
Guess 2: Traffic signals (0%)
Guess 3: Right-of-way at the next intersection (0%)
Guess 4: No passing for vehicles over 3.5 metric tons (0%)
Guess 5: Ahead only (0%)
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Children crossing

Guess 1: Children crossing (99%)
Guess 2: Bicycles crossing (1%)
Guess 3: Ahead only (0%)
Guess 4: Go straight or right (0%)
Guess 5: Turn left ahead (0%)
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAP8AAAD8CAYAAAC4nHJkAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz
AAALEgAACxIB0t1+/AAAG0BJREFUeJztnV2MJFd1x/+nqrvnez9m17ve7DrYICsBoWDQyEJyhAgk
yEFIBikg/ID8YLEowlKQyIPlSMGR8gBRAPFEtMQWJiIYB4OwIivBsogsXgyLY/zB8mHMBpYde23v
zs739EedPHQ7Gc/W/0xPz0z1eu//J622p07furdu1enqvv8655i7QwiRHtmwByCEGA5yfiESRc4v
RKLI+YVIFDm/EIki5xciUeT8QiSKnF+IRJHzC5Eote00NrObAXwJQA7gn939s2FntZrXGyPEOsiT
hjZAG8CCZnnGPw+zvNzW6RS0TafgtuiQLRrkQHMV7O1yecjT+NxH56UoOtTGn2DlBx3NvUftBp5H
0t8Al3dzbQ3tdruvljbo471mlgP4BYA/A3AGwI8A3OruP2VtxsYn/I1/8IeltiJwIHYyQgcJTPWM
G/ftmaC2cWI7P79E2ywtcBuCz4U8r/NmwcXO5iQ6y95pB8bAEQb48CoCB0eN3RiAqfEpaltdvkht
rfZKuSHw1Dy4PiJ/yUIbNQGWl/cVXN9sfn9+6lksLy315fzb+dp/I4Dn3P15d28CuB/ALdvYnxCi
Qrbj/EcB/Hbd32d624QQrwO285u/7KvFJV9uzOw4gOMAUK83ttGdEGIn2c6d/wyAa9b9fQzA2Y1v
cvcT7j7j7jN5bVvri0KIHWQ7zv8jANeb2XVm1gDwUQAP7cywhBC7zcC3Yndvm9kdAP4TXanvXnd/
dpNWMLLuHEk5jGi136Kl9IACLWpbXnypdHt7ha+W586Pq93hS8CRpDSI7BWtYBeDKabw4N5R0OEH
K+lcxMDK0hq11bPgMia2dqfJxxHNfY2rMAgUqyKQfJlIEMqK7BrYgni3re/h7v4wgIe3sw8hxHDQ
E35CJIqcX4hEkfMLkShyfiESRc4vRKJU+tSNmaGWkyCGICiC2WqBfIVAYgtiVTC/GElKZIwdvsPy
o+0SiZF5MMZOJ4hiI1JfFFyYRTJgGP0W6ErknEXn2cDlt1oWSGXtVWprN8sDe7JAsmsHcU5RXFIR
nBcLJM4wgJPgRBfdSpie7vxCJIqcX4hEkfMLkShyfiESRc4vRKJUHmPrZNUzC1aOMxbEECxhZ8ES
aptHnaAdBGegVp6PoFPw5eFIkIiWeVvBknMU8MECZ6L5iFSHwiPr1tOJ5cGqdxas9neai9SWB2Mc
YVd4FBSWcSUgmo5oRT/KeebkDNDrPmArwoHu/EIkipxfiESR8wuRKHJ+IRJFzi9Eosj5hUiUiqU+
A4h8MTnOK+U0V8oDN9aaPKAj0jyi3H95JL+RwI1O0Cac4AGCmYBNcruRocRBRME9ILIFsOHXgrnq
NLl0WLS49BmVROsQWXdklIdc1Ue51NcKtL64glSo+W5xO2guzK2gO78QiSLnFyJR5PxCJIqcX4hE
kfMLkShyfiESZVtSn5mdBrCAbnhX291n4ga81NTi4jJt5m2Sly7I0zc2PkZtoUQYRPxRW9DGw5Ji
QekqPopNcr4RKSoc42BSnwf7ZEOMpNS9k3uo7cLLr/C+Cj7Gxkj5dVBrcKmvHdQNi+Q8JisCQC3I
Geg0qWSwP9JkK/kAd0Ln/xN3f3kH9iOEqBB97RciUbbr/A7ge2b2YzM7vhMDEkJUw3a/9t/k7mfN
7BCAR8zsZ+7+2Po39D4UjgNAvVGeCUcIUT3buvO7+9ne/+cAfAfAjSXvOeHuM+4+U6sHtc2FEJUy
sPOb2YSZTb36GsD7ADyzUwMTQuwu2/nafxjAd3rSRw3Av7r7f0QN3B0tEp3FKmFFRJFqq0ECzFjN
CxKJMnklGEfEwHJeYMzI53mYwDMYCBe9EB5AbuVSWu78krMWP6M1UoYMALIg0m5ibLLc0OD3vYU1
XrKtCAqwRclOW50WtbFzw2RxAKjXyucxjCzcwMDO7+7PA3jboO2FEMNFUp8QiSLnFyJR5PxCJIqc
X4hEkfMLkSjVJvB0wEktvFYUdUbkt0gKKYKae4GaF8KSJmYZl3+i0n9hX2EtuShSkO6QtwnmI5Jg
w2m08ge6GiNTtElrZY73FZ60IPFnUR7BOTHKIwibgZzXXAvqMgbXgQXCND3XwfXdaZfXNYznacPu
+36nEOKKQs4vRKLI+YVIFDm/EIki5xciUSou1xWsmOdB4An9jAraBCWc4pCgQaSAqE1UWmvQUlhB
f2TlOAreicp1Zc4DUgrjq+y1Rvml1RgpX6UGgKWFJWrzMK8eNWFhfrF0exsjtE196ipqyws+xk6T
56EMBBo4CQiqj/AQ+GPT06Xbf/ncr3hHG8fU9zuFEFcUcn4hEkXOL0SiyPmFSBQ5vxCJIucXIlEq
lfoMAFP0ovxnNoDUF8lhg+bHG6RJFmg8HmX/oyWc4oCmDsl1Fx1VlLcwkkWnxnggy9RY+RiXLpyl
bVrLXEarGb9Us+DocnJoyxe55Djm/LgmRyeobdF5GTjP+DyOjJSntK8H8vcrc+Xly9odHni0Ed35
hUgUOb8QiSLnFyJR5PxCJIqcX4hEkfMLkSibSn1mdi+ADwA45+5v7W2bBvBNANcCOA3gI+5+oa8e
ifTCykwBPOdeEUTMhVFsQa41D6Pwtt5XJ0jiF0UyRhSdIMKNbI+Oy6P8chmvrJwX3NaZLy95tXy+
PMoOABrBeYmOOUpCyKY4Mx6BZ2svUNtEIPWtBrkEm0HkZLNV3s6DiEqvlc/VVvJT9nPn/yqAmzds
uxPAo+5+PYBHe38LIV5HbOr87v4YgPMbNt8C4L7e6/sAfHCHxyWE2GUG/c1/2N1nAaD3/6GdG5IQ
ogp2/fFeMzsO4DgA1Ov8N6IQoloGvfO/aGZHAKD3/zn2Rnc/4e4z7j5TIzXFhRDVM6jzPwTgtt7r
2wB8d2eGI4Soin6kvm8AeDeAg2Z2BsBnAHwWwANmdjuA3wD4cH/dOQqiiw1SQiuSr8J2Qf7OPA9k
QNIwzKeZRVMcRfxFxzZA5GEQXVi0eSRYPR/jO3Uue82zZJzGf/pFR9wY5+1GRnmiy7XV8nG01lZo
mzw4ZZ1Ashsb50lBizaXATtExvTg3txssWuxf5/Y1Pnd/VZiem/fvQghLjv0hJ8QiSLnFyJR5PxC
JIqcX4hEkfMLkSiVPnXjHkTADaL1RTXmAmkrj5JqkgSYAJffLIhGqwW6USeoJ2hhclJqolJPKAFF
yU4LLolZziXCvFZu6zT5cWVBBOHUviPU1hjn879y9tel2w28zcUFnohzoj5FbVPT+6itucyjGVut
8gjDVjvQpNl55i0uQXd+IRJFzi9Eosj5hUgUOb8QiSLnFyJR5PxCJErFAfYGFpEWRbFlREqLat2F
NfKisL5QLCm3daL6aIGMVgtq7kXHVgTFAZmUGkl9E6Oj1HZgistvixdmqa21XC4RFuCRb62cj6O+
h0t9o3v4PE6sLZRuPz/Lk3RmFkiHF7hkV88mqW1ynB9bOyuvG9jsBD7BZO4dTuAphLgCkfMLkShy
fiESRc4vRKLI+YVIlIpX+z1YdY4S4bFgFb5qXzj/XIsy4GXBSjobuwWr9hYF1ISlvPiKc6fg6gIb
fxR8NDnKV6mX5zbWa1lv4yWv6l7eXzsorTV5kAfGNPZw1WHFeFmrvYePlW7vBEEzc+e4itEITufq
3EvUVrNpahvJy4+t1Q7KdbF8krTFpejOL0SiyPmFSBQ5vxCJIucXIlHk/EIkipxfiETpp1zXvQA+
AOCcu7+1t+1uAB8H8Kq2cZe7P7zZvhxAh0gUeSSXEfUtiN0ZuNxVJNtlZJcW5BL0IDijRco0AZvk
8AOXqVh+wjznATWrK1xSWpkjZbcAGLj81ibzf+Awr+a+7zCXw0CCXwAgj+a/KD/uyf28r+WFl/n+
lvk4OgU/ZysrXBatTZRLnDULSnyRXJNbKeTWz53/qwBuLtn+RXe/ofdvU8cXQlxebOr87v4YAP6k
hxDidcl2fvPfYWZPmdm9ZrZ/x0YkhKiEQZ3/ywDeBOAGALMAPs/eaGbHzeykmZ0Mk14IISplIOd3
9xfdvePdB4y/AuDG4L0n3H3G3WfyqPC5EKJSBnJ+M1ufU+lDAJ7ZmeEIIaqiH6nvGwDeDeCgmZ0B
8BkA7zazG9BV704D+EQ/nZkZ8np5tFco23m5rJEFslyrzWWSPK9TGy0nBqDGcgkGjTw6ruCgo59I
Wc6PuyBy0whX5bC2dIHaRmp8jM1WUBJtZLx8fxNcYisyLkfWgnPmgWTKLixrjNEmjTG+hNVamxto
HEGwKNjpnBzjJ21+iUmH/cf1ber87n5ryeZ7+u5BCHFZoif8hEgUOb8QiSLnFyJR5PxCJIqcX4hE
qfSpmywzjJGyRe3WGm3XabMyWTwaLQh8g1tw2IH81iJJH7NAXgljCyP9J/hY7gTGSSKxjed8Qi62
yktaAUDR4kdgweXTGClPCjo+zqU+I4ksAcCDZKdBTlBKfWSC2vbsu5raXlkoL0MGANbhkXuri8Ec
k/O5/2ouORYkojIqU3fJe/t+pxDiikLOL0SiyPmFSBQ5vxCJIucXIlHk/EIkSqVSX+GO1bVySa8T
1CVjepkNWI+P1wsE2kE0IJP0osi9SM6zQJYJclIiq/MIt7HR8si4tfPneF8tLqPVEERABgk8p/bs
Ld0+Ph7U3IvqGga2LErISkxFcNL27j/Ih7FykdrOzXJbHszjCpEBa3M8ytEbrL5i//dz3fmFSBQ5
vxCJIucXIlHk/EIkipxfiESpdLXfC0e7Wb6qH6SlQ5uUvMpQnlMPiHOmRav9UV5AtjjPSmRtNg4L
IlIawYTU8qAs1MXyHHPNhUU+kGDlu5nxVeo9B49S276rj5RuX2zzAK7lYLG/bvxc14Nrp0nyK7aC
+16UEnDy0FXUtrTKFZXluVVqy8j1uDLHA4VG906VGwIF7JJ++36nEOKKQs4vRKLI+YVIFDm/EIki
5xciUeT8QiRKP+W6rgHwNQBXo5sZ74S7f8nMpgF8E8C16Jbs+oi787pP6MpeOYlYsSDbXb1WLvNY
IPXBgyR+gZwXBenkpF1RcG2o0+Zlt6KyYQgKGo82+FwtzZdLffVgPpqB9Dk6yQJIgL2B7NW08jn5
9dnTtM2S8/nYP87z2V29vzxvIQD87PmflRtG99A2vx9ImFm9PAclAEzsP0ZtrZVZaitWyuXPWhHI
g81XyM76r4Tdz52/DeDT7v5mAO8E8EkzewuAOwE86u7XA3i097cQ4nXCps7v7rPu/kTv9QKAUwCO
ArgFwH29t90H4IO7NUghxM6zpd/8ZnYtgLcDeBzAYXefBbofEAAO7fTghBC7R9+P95rZJIAHAXzK
3efDnPOvbXccwHEAqAdJKIQQ1dLXnd/M6ug6/tfd/du9zS+a2ZGe/QiA0geb3f2Eu8+4+0xeqzSU
QAgRsKnzW/cWfw+AU+7+hXWmhwDc1nt9G4Dv7vzwhBC7RT+34psAfAzA02b2ZG/bXQA+C+ABM7sd
wG8AfHjzXRmMRGe5c4mi0ym3RdFttSAqrgikrYJEgcXtgpJWgazYDKLHpib2ceMaj9AzKunxMeYN
nldvZJKXtQKRYAGgsPJzZgUvdzX/4svUdug6nlcvkolrpL/zs/O8zYHfo7YsH6O2xhgfYz7Cz5mv
Nku3F5HUR352mwUS9wY2dX53/wF4Psz39t2TEOKyQk/4CZEocn4hEkXOL0SiyPmFSBQ5vxCJUulT
NwYgIxJcqxXJZSSqL4rOG6CEE7DJpyGR+qJEnFFJrrFxXo5pfJTvc3GOy1R8KFzOqzXKS2sBwMTe
A9SW1bnU1yFSXx4k6RwLJNjo2dAoIWutKJe+JqIo0sDmgW10nEf87TvAowiXF14o3W4dLtvNX1wq
3d6Jso9uQHd+IRJFzi9Eosj5hUgUOb8QiSLnFyJR5PxCJEq1AfZmVOqrB1JUh0oe/SUUuWR/QeQe
G1/cG5dkxoPItz0j3LZ6/nfU1l7hNdzcyudx1biseNX0NdQ2sZdHqjU7PEKPJXvpNwnMRiI5L1AP
qS0aR0HkQSCOICwC2+QBHqU5vVqenPSVsyRJJwA4c93+51d3fiESRc4vRKLI+YVIFDm/EIki5xci
USpd7Xd3tFjgQbACz/L+RZ9dRVCeKrKFVb7oCjYPO2mAr7J3LvC8bmtz5YEbAFAPVr5XrDwf3OTB
w7TNvkN8JboZ5Fb0aOWeTGQUBBXEQIXnDKQ0WLe/Vvn+ivLtAOBF+RwCQNHhLlME2ak7ztWsfVe9
oXT7Ghd1MH/+PDf2ie78QiSKnF+IRJHzC5Eocn4hEkXOL0SiyPmFSJRNpT4zuwbA1wBcjW4Eywl3
/5KZ3Q3g4wBe6r31Lnd/eLP9FSRIJ49y7gXSFiMLdKMwECToiwarRNNo49S0FJTd6tDADSAHl6n2
7inPIzd9Fc/Tl2eBnBfcH+KzsvV8h0ECwjCgJg9KVO3fX15ubH7updLtAHD2pVlqO3ysXJYDgAZ4
oFYj51IfsqnSzZP7r6JNlhfL8zhuJW6qH52/DeDT7v6EmU0B+LGZPdKzfdHd/7H/7oQQlwv91Oqb
BTDbe71gZqcAHN3tgQkhdpct/eY3s2sBvB3A471Nd5jZU2Z2r5mVByULIS5L+nZ+M5sE8CCAT7n7
PIAvA3gTgBvQ/WbwedLuuJmdNLOT7Tb/bSmEqJa+nN+6D68/CODr7v5tAHD3F9294+4FgK8AuLGs
rbufcPcZd5+pBc8+CyGqZVPnt+4S9z0ATrn7F9ZtP7LubR8C8MzOD08IsVv0cyu+CcDHADxtZk/2
tt0F4FYzuwFdTec0gE/01SNRbKKcakYiuqLSSUVn65LdoOR58HMm43IegnZRPjjLxqitlpeX1+o0
+ef86BS3dYKpCvPZ0ag+Loe5RyXWgvtUEDE3PX1t6fa1VV5a6/lZnj9xOfj2+vuHeS7EPOPtsrz8
uOvjk7RNfaRcwozmdyP9rPb/AOVZATfV9IUQly96wk+IRJHzC5Eocn4hEkXOL0SiyPmFSJRqn7px
HrgVRtMNECHmgTTkYUkjvs96Vr7PA5Ncalq88AK1rS3wJJ0Iymut5eUyDwBM77+udPvoPh5d2AqS
dBqRoQAgC5KuOrOFoYBBJGYgAzq4bJcRie3IEX7pL60uUNvLL3AZcHWcJ0KdCMp1sUMbmeBS38S+
8oi/LD9N21zy3r7fKYS4opDzC5Eocn4hEkXOL0SiyPmFSBQ5vxCJUq3UZwBVjqJabEQfKgJpKIqK
y4lkBwBFwfc5kpdH061d5EXV1i6uUttoEIG1mvH5mDrMkyaN7idjjOrZUQuQBRJbnvEahazeXSTp
RgSqYpi1kiVydefXwOHpg9S2MhdIt4tr1OYH+LkuSALSVuATB48cKd1e+yk/JxvRnV+IRJHzC5Eo
cn4hEkXOL0SiyPmFSBQ5vxCJUqnUZ+CqTFg/LxSjSF+RpNThUWz1jEeIebtcrlmc53KeeyCHBfLb
gUPT1LYnqLvnVi6xhQkwo/kNpL52oM6y/qKkqxhwjFE+1sLL59jyQHoLZUXuMh7UV7SwHTlnwTF3
BvCJjejOL0SiyPmFSBQ5vxCJIucXIlHk/EIkyqar/WY2CuAxACO993/L3T9jZtcBuB/ANIAnAHzM
2bJlD4ejIKvw0Yo+Lf00QN6/zWyZ8yAd8/LAjXrOV+2bQSWvWp3n4quNlOdoAwCv8dxuIEFL0cpx
vJLO56rdbvF2LDApzK0YKQFcWmDXx6t7Ld1dcFwFtr6/zcZRFNzG4syiICgnEsdWwqb6ufOvAXiP
u78N3XLcN5vZOwF8DsAX3f16ABcA3L6FfoUQQ2ZT5/cur1abrPf+OYD3APhWb/t9AD64KyMUQuwK
ff3mN7O8V6H3HIBHAPwKwJz7/+V8PgPg6O4MUQixG/Tl/O7ecfcbABwDcCOAN5e9raytmR03s5Nm
drLTDn4ACyEqZUur/e4+B+C/ALwTwD77/2cWjwE4S9qccPcZd5/Jg9rmQohq2dT5zewqM9vXez0G
4E8BnALwfQB/0XvbbQC+u1uDFELsPP3cio8AuM/McnQ/LB5w9383s58CuN/M/h7AfwO4Z/NdGZWc
ojgcZssDiaoRSDnjo7wU1tgID/hYufhK6fbmKg/sQRQo1NhDbY3x8hxtAJDX+D6LonwsFkW/BHNP
UuD19rl1+W3AVHxAIL+Fsi6RFkO5N7h2onFEcxXlqGTNot2FwUd9sqnzu/tTAN5esv15dH//CyFe
h+gJPyESRc4vRKLI+YVIFDm/EIki5xciUWzQ8kkDdWb2EoD/6f15EMDLlXXO0Thei8bxWl5v43iD
u/OQ0HVU6vyv6djspLvPDKVzjUPj0Dj0tV+IVJHzC5Eow3T+E0Psez0ax2vROF7LFTuOof3mF0IM
F33tFyJRhuL8Znazmf3czJ4zszuHMYbeOE6b2dNm9qSZnayw33vN7JyZPbNu27SZPWJmv+z9v39I
47jbzH7Xm5Mnzez9FYzjGjP7vpmdMrNnzeyvetsrnZNgHJXOiZmNmtkPzewnvXH8XW/7dWb2eG8+
vmlmjW115O6V/gOQo5sG7I0AGgB+AuAtVY+jN5bTAA4Ood93AXgHgGfWbfsHAHf2Xt8J4HNDGsfd
AP664vk4AuAdvddTAH4B4C1Vz0kwjkrnBN1o3sne6zqAx9FNoPMAgI/2tv8TgL/cTj/DuPPfCOA5
d3/eu6m+7wdwyxDGMTTc/TEA5zdsvgXdRKhARQlRyTgqx91n3f2J3usFdJPFHEXFcxKMo1K8y64n
zR2G8x8F8Nt1fw8z+acD+J6Z/djMjg9pDK9y2N1nge5FCODQEMdyh5k91ftZsOs/P9ZjZteimz/i
cQxxTjaMA6h4TqpImjsM5y9LUDIsyeEmd38HgD8H8Ekze9eQxnE58WUAb0K3RsMsgM9X1bGZTQJ4
EMCn3H2+qn77GEflc+LbSJrbL8Nw/jMArln3N03+udu4+9ne/+cAfAfDzUz0opkdAYDe/+eGMQh3
f7F34RUAvoKK5sTM6ug63Nfd/du9zZXPSdk4hjUnvb63nDS3X4bh/D8CcH1v5bIB4KMAHqp6EGY2
YWZTr74G8D4Az8StdpWH0E2ECgwxIeqrztbjQ6hgTqybYPAeAKfc/QvrTJXOCRtH1XNSWdLcqlYw
N6xmvh/dldRfAfibIY3hjegqDT8B8GyV4wDwDXS/PrbQ/SZ0O4ADAB4F8Mve/9NDGse/AHgawFPo
Ot+RCsbxx+h+hX0KwJO9f++vek6CcVQ6JwD+CN2kuE+h+0Hzt+uu2R8CeA7AvwEY2U4/esJPiETR
E35CJIqcX4hEkfMLkShyfiESRc4vRKLI+YVIFDm/EIki5xciUf4Xsm9EIdbKIR0AAAAASUVORK5C
YII=
"
>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Road narrows on the right

Guess 1: Keep right (100%)
Guess 2: Speed limit (60km/h) (0%)
Guess 3: End of speed limit (80km/h) (0%)
Guess 4: Right-of-way at the next intersection (0%)
Guess 5: Speed limit (80km/h) (0%)
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Analyze-Performance">Analyze Performance<a class="anchor-link" href="#Analyze-Performance">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Output-Top-5-Softmax-Probabilities-For-Each-Image-Found-on-the-Web">Output Top 5 Softmax Probabilities For Each Image Found on the Web<a class="anchor-link" href="#Output-Top-5-Softmax-Probabilities-For-Each-Image-Found-on-the-Web">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For each of the new images, print out the model's softmax probabilities to show the <strong>certainty</strong> of the model's predictions (limit the output to the top 5 probabilities for each image). <a href="https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k"><code>tf.nn.top_k</code></a> could prove helpful here.</p>
<p>The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.</p>
<p><code>tf.nn.top_k</code> will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.</p>
<p>Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. <code>tf.nn.top_k</code> is used to choose the three classes with the highest probability:</p>

<pre><code># (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])</code></pre>
<p>Running it through <code>sess.run(tf.nn.top_k(tf.constant(a), k=3))</code> produces:</p>

<pre><code>TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))</code></pre>
<p>Looking just at the first row we get <code>[ 0.34763842,  0.24879643,  0.12789202]</code>, you can confirm these are the 3 largest probabilities in <code>a</code>. You'll also notice <code>[3, 0, 5]</code> are the corresponding indices.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Project-Writeup">Project Writeup<a class="anchor-link" href="#Project-Writeup">&#182;</a></h3><p>Once you have completed the code implementation, document your results in a project writeup using this <a href="https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md">template</a> as a guide. The writeup can be in a markdown or pdf file.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<blockquote><p><strong>Note</strong>: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "<strong>File -&gt; Download as -&gt; HTML (.html)</strong>. Include the finished document along with this notebook as your submission.</p>
</blockquote>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<hr>
<h2 id="Step-4-(Optional):-Visualize-the-Neural-Network's-State-with-Test-Images">Step 4 (Optional): Visualize the Neural Network's State with Test Images<a class="anchor-link" href="#Step-4-(Optional):-Visualize-the-Neural-Network's-State-with-Test-Images">&#182;</a></h2><p>This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.</p>
<p>Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the <a href="https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81">LeNet lab's</a> feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.</p>
<p>For an example of what feature map outputs look like, check out NVIDIA's results in their paper <a href="https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/">End-to-End Deep Learning for Self-Driving Cars</a> in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.</p>
<p><figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">### Visualize your network&#39;s feature maps here.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>

<span class="c1"># image_input: the test image being fed into the network to produce the feature maps</span>
<span class="c1"># tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer</span>
<span class="c1"># activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output</span>
<span class="c1"># plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry</span>

<span class="k">def</span> <span class="nf">outputFeatureMap</span><span class="p">(</span><span class="n">image_input</span><span class="p">,</span> <span class="n">tf_activation</span><span class="p">,</span> <span class="n">activation_min</span><span class="o">=-</span><span class="mi">1</span><span class="p">,</span> <span class="n">activation_max</span><span class="o">=-</span><span class="mi">1</span> <span class="p">,</span><span class="n">plt_num</span><span class="o">=</span><span class="mi">1</span><span class="p">):</span>
    <span class="c1"># Here make sure to preprocess your image_input in a way your network expects</span>
    <span class="c1"># with size, normalization, ect if needed</span>
    <span class="c1"># image_input =</span>
    <span class="c1"># Note: x should be the same name as your network&#39;s tensorflow data placeholder variable</span>
    <span class="c1"># If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function</span>
    <span class="n">activation</span> <span class="o">=</span> <span class="n">tf_activation</span><span class="o">.</span><span class="n">eval</span><span class="p">(</span><span class="n">session</span><span class="o">=</span><span class="n">sess</span><span class="p">,</span><span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">x</span> <span class="p">:</span> <span class="n">image_input</span><span class="p">})</span>
    <span class="n">featuremaps</span> <span class="o">=</span> <span class="n">activation</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">3</span><span class="p">]</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">plt_num</span><span class="p">,</span> <span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">15</span><span class="p">,</span><span class="mi">15</span><span class="p">))</span>
    <span class="k">for</span> <span class="n">featuremap</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">featuremaps</span><span class="p">):</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">subplot</span><span class="p">(</span><span class="mi">6</span><span class="p">,</span><span class="mi">8</span><span class="p">,</span> <span class="n">featuremap</span><span class="o">+</span><span class="mi">1</span><span class="p">)</span> <span class="c1"># sets the number of feature maps to show on each row and column</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;FeatureMap &#39;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">featuremap</span><span class="p">))</span> <span class="c1"># displays the feature map number</span>
        <span class="k">if</span> <span class="n">activation_min</span> <span class="o">!=</span> <span class="o">-</span><span class="mi">1</span> <span class="o">&amp;</span> <span class="n">activation_max</span> <span class="o">!=</span> <span class="o">-</span><span class="mi">1</span><span class="p">:</span>
            <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">activation</span><span class="p">[</span><span class="mi">0</span><span class="p">,:,:,</span> <span class="n">featuremap</span><span class="p">],</span> <span class="n">interpolation</span><span class="o">=</span><span class="s2">&quot;nearest&quot;</span><span class="p">,</span> <span class="n">vmin</span> <span class="o">=</span><span class="n">activation_min</span><span class="p">,</span> <span class="n">vmax</span><span class="o">=</span><span class="n">activation_max</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s2">&quot;gray&quot;</span><span class="p">)</span>
        <span class="k">elif</span> <span class="n">activation_max</span> <span class="o">!=</span> <span class="o">-</span><span class="mi">1</span><span class="p">:</span>
            <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">activation</span><span class="p">[</span><span class="mi">0</span><span class="p">,:,:,</span> <span class="n">featuremap</span><span class="p">],</span> <span class="n">interpolation</span><span class="o">=</span><span class="s2">&quot;nearest&quot;</span><span class="p">,</span> <span class="n">vmax</span><span class="o">=</span><span class="n">activation_max</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s2">&quot;gray&quot;</span><span class="p">)</span>
        <span class="k">elif</span> <span class="n">activation_min</span> <span class="o">!=-</span><span class="mi">1</span><span class="p">:</span>
            <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">activation</span><span class="p">[</span><span class="mi">0</span><span class="p">,:,:,</span> <span class="n">featuremap</span><span class="p">],</span> <span class="n">interpolation</span><span class="o">=</span><span class="s2">&quot;nearest&quot;</span><span class="p">,</span> <span class="n">vmin</span><span class="o">=</span><span class="n">activation_min</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s2">&quot;gray&quot;</span><span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">activation</span><span class="p">[</span><span class="mi">0</span><span class="p">,:,:,</span> <span class="n">featuremap</span><span class="p">],</span> <span class="n">interpolation</span><span class="o">=</span><span class="s2">&quot;nearest&quot;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="s2">&quot;gray&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
    </div>
  </div>
</body>

 


</html>

{:/}

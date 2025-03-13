Folder with the latex code to generate the different taxonomies that are shown in the main document.


To generate each graph, you need to compile the latex code with the following command:

```bash
pdflatex -output-directory=build <file>
```

Where `<file>` is the name of the file you want to compile, and there is a local folder named `build`.
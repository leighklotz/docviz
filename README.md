# Output
![docs/examples/spam_topic_umap_plot.png](docs/examples/spam_topic_umap_plot.png)

# install
```
$ for i in $(grep '^From ' caughtspam-20250601 | awk '{print $2}' | sort -u); do cat caughtspam-20250601 | grep "^From $i" > "$i.mbox"; done
$ cd ..
$ mkvenv
$ pip install -r requirements.txt
./run_top2vec.py docs/
```

For output see [docs](docs).



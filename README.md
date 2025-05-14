# Wait is self-attention Bidirectional maked ?

Okay, I’m thinking that I need to embed citations in a clean and minimal way directly inside the docstrings of the code. I'll quickly search for the key differences between BERT (Bidirectional) and GPT (Causal), gather a few strong sources, and use them in the docstrings to provide that context. Since the user wants minimal yet descriptive docstrings, I will enhance the existing ones slightly, ensuring they explain the differences in training patterns while maintaining the requested brevity. I'll be strategic with embedding citations. Should work!

I’m thinking of creating a module-level docstring at the top of the code to clarify the difference between Bidirectional and Causal attention models. I'll define Bidirectional (BERT) in terms of its masked language model, and Causal (GPT) in terms of autoregressive models. The goal, I believe, is to show that causal attention is better for swift code synthesis while keeping the docstrings minimal and logical.

For citation embedding, I'll reference multiple sources to back up the explanation. Then I'll add the code with the updated docstring.
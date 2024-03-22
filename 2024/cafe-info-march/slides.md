class: middle

# HTML presentations 🧑🏼‍🏫️

#### Café info – 21 mars 2024
#### [Alexandre Boucaud](mailto:aboucaud@apc.in2p3.fr)

.footnote[Use arrow keys <kbd>←</kbd> and <kbd>→</kbd> to navigate between slides]

---

## Lists and tables

1. first thing 
2. second thing

- one thing
- another thing

| **name** | **size** | **age** |
| -------- | -------: | ------: |
| Paul     |       90 |       3 |
| Anna     |      140 |      10 |

---
class: center

### A native (large) image

![](../img/cafe-info-logo.png)

---
class: center

### Same image scaled

<img src="../img/cafe-info-logo.png" width=45%/>

Code written directly in HTML for more display options

---
class: middle
## Using both sides

.left-column[

### Code examples
```python
# This is a code example in Python
# with natural syntax highlighting

import numpy as np
from numpy.random import standard_normal

def model(x):
    return x + 3 * np.sin(x)

if __name__ == "__main__":
    sigma = 0.2
    x_plot = np.arange(0, 10, 0.1)
    noise = sigma * standard_normal(x_plot.shape)
    # emphasize a given line
*   data = model(x_plot) + noise
    print(data)

```
.center[
.caption[pretty and easy to copy/paste]
]
]

.right-column[
### Animations
![](../img/presentation.gif)  
.center[
.caption[supported natively]
]
]

---
class: middle,center

### That's it for today !

This presentation was written in .green[Markdown] using the [remark.js](https://remarkjs.com/) framework.

The presentation is available at  
https://aboucaud.github.io/slides/2024/cafe-info-march

and the markdown source code can be found [here][mdsource].

[mdsource]: https://raw.githubusercontent.com/aboucaud/slides/master/2024/cafe-info-march/slides.md

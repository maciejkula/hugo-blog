+++
title = "Don't use explicit feedback recommenders"
author = ["maciej"]
date = 2018-07-19T19:02:00+01:00
lastmod = 2018-07-19T19:02:28+01:00
categories = ["engineering"]
draft = false
weight = 2002
+++

Back in January, I gave a talk at the [London RecSys Meetup](https://www.meetup.com/RecSys-London/events/245357880/) about why explicit feedback recommender models are inferior to implicit feedback models in the vast majority of cases.

The key argument is that what people choose to rate or not rate expresses a more fundamental preference than what the ratings is. Ignoring that preference and focusing on the gradations of preference _within_ ranked items is the wrong choice.

The slides are below, and you can watch the recording [here](https://skillsmatter.com/skillscasts/11375-explicit-vs-implicit-recommenders). If you are interested in confirming this for yourself, have a look at my [explicit-vs-implicit experiment](https://github.com/maciejkula/explicit-vs-implicit).

<script async class="speakerdeck-embed" data-id="c528f4ca53ec44969d34478b41806698" data-ratio="1.77777777777778" src="//speakerdeck.com/assets/embed.js"></script>

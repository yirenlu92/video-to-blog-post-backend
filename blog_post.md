

## Creating our own Kubernetes and Docker to run our data infrastructure

When I started Modal Labs, I wanted to build better tools for data engineers and data scientists. However, I quickly realized that to achieve this goal, I needed to create a lot of underlying infrastructure. This led me down a deep rabbit hole of building our own versions of Kubernetes and Docker to run our data infrastructure.

Our primary objective was to take code that exists on a local computer, spin up an arbitrary container in the cloud from a pool of workers, and then run that container. We wanted to do this so fast that it would feel almost like local development, or even better.

The challenge was that Docker, the standard tool for container management, was too slow for our needs. Pulling down a Docker image can take anywhere from 10 seconds to several minutes, depending on its size. This latency was unacceptable for our goal of near-instant container deployment.

To optimize this process, we made several key observations:

1. Most of the files in a container image are never actually read during execution.
2. When starting a container, only a small subset of files (about 3% in the case of scikit-learn) are accessed.
3. Even unrelated images often have a significant overlap in the files they contain.

Based on these insights, we developed a custom solution that includes:

1. A content-addressing system to efficiently store and cache files.
2. A custom file system implemented in FUSE to handle file access.
3. A pre-computation step to build indexes for container images.
4. A custom implementation of container building, similar to Docker but optimized for our needs.

This approach allows us to start containers in the cloud in about a second, running custom images with user code inside. We've essentially created a function-as-a-service platform where users can write Python functions, decorate them, and deploy them to the cloud almost instantly.

Our system is particularly beneficial for GPU workloads, where the ability to scale quickly and efficiently is crucial due to the high cost of GPU instances. We've optimized the entire process, from booting containers to loading large models, to minimize cold start times.

While building this infrastructure in-house might not be the right choice for most companies, it has allowed us to create a platform that's better suited for data teams than existing solutions like Kubernetes. Our approach eliminates the need for multiple layers of abstraction and provides a more streamlined experience for data engineers and scientists.




## Building your own Kubernetes and Docker

As the founder of Modal, a company providing data infrastructure in the cloud, I've been on a journey to build better tools for data engineers and data scientists. This journey led me down a deep rabbit hole of infrastructure development. Today, I want to share with you the technical aspects of building our own container and orchestration system, similar to Kubernetes and Docker, but optimized for our specific needs.

Our goal was to create a system that allows users to write code locally, spin up a container in the cloud, and run that code inside an arbitrary container - all within about a second. We wanted this process to feel almost like local development, or even better.

To achieve this, we had to overcome several challenges:

1. **Fast container startup**: Traditional Docker image pulls are too slow for our needs, often taking minutes for large images.

2. **Efficient file access**: We needed to optimize file access within containers to reduce latency.

3. **Quick image building**: Building container images needed to be faster than traditional Docker builds.

4. **Resource management**: We had to maintain a pool of worker instances in the cloud, scaling them up and down as needed.

5. **Workload scheduling**: We needed a mechanism to assign work to cloud workers efficiently.

In the following sections, I'll dive into how we tackled each of these challenges, ultimately creating a function-as-a-service platform that can deploy Python functions to the cloud in about a second, with automatic scaling capabilities.




## I'm Erik Bernhardsson

I'm the founder of Modal Labs, a company that provides data infrastructure in the cloud. Before starting Modal, I spent most of my career working with data. One of my notable achievements was building the music recommendation system at Spotify. I've also created open-source projects like Luigi, a workflow scheduler that was widely used at one point, and Annoy, a vector database that gained some popularity in its time.

For those interested in following my work and thoughts:

- I tweet occasionally: @bernhardsson
- I blog very occasionally: https://erikbern.com

My experience in the data field, particularly as a CTO for many years, has given me unique insights into the challenges faced by data engineers and scientists. This background has been instrumental in shaping the vision for Modal Labs, where we're working on creating better tools and abstractions for data professionals.




## All just wanted to make data teams more productive!

When I started thinking about better tools for data engineers a couple of years ago, my goal was simple: I wanted to make data teams more productive. After years of working with data and banging my head against AWS docs, I figured we could build better abstractions. I wanted to create something that would make me happy when writing code, productionizing stuff, scheduling things, and using GPUs.

The key areas I focused on were:

1. How to productionize jobs
2. How to scale things out
3. Scheduling things
4. How to use GPUs and other hardware

These challenges are like a deep rabbit hole, and addressing them required building a lot of infrastructure. But it's important to remember that technology is a means to an end. There's no point in building tech if you don't have a goal (although, admittedly, it can be fun to build tech for its own sake).

When thinking about developer productivity, I like to consider the pseudocode of writing code as a nested set of for loops. The innermost loop is where you're writing code, running it, and getting immediate feedback in seconds. As you move outward, the feedback loops get longer:

- Running unit tests or scripts locally: minutes
- Creating pull requests and waiting for CI/CD: hours
- Deploying to production and waiting for scheduled jobs: days

These long feedback loops can ruin the joy of writing code. It's not fun to wait. Contrast this with front-end engineers, who have amazing tools that provide almost immediate gratification. Backend engineers often have compiler feedback and unit tests, but data teams often face frustratingly long wait times for cron jobs, model training, or scaling out operations.

My goal was to compress these feedback loops by taking concerns from the outermost loops and putting them back into the innermost loops. One of the biggest issues is that infrastructure happens in the outermost loop, causing things to break in production in ways they didn't break locally.

To address this, we needed to build a system that could:

1. Take code that exists on a local computer
2. Spin up an arbitrary container in the cloud out of a pool of workers
3. Run that container with the local code

By solving these challenges, we aimed to create a development experience for data teams that feels almost like local development, but with the scalability and power of the cloud. This approach allows us to build better tools for data engineers and data scientists, making them more productive and bringing back the joy of writing code for data applications.




## What do I mean by eng productivity?

When I think about developer productivity, I visualize it as a set of nested for-loops of writing code. This concept is best understood by examining the workflow of a typical software engineer.

At the innermost loop, you're writing code and getting immediate feedback within seconds. You might encounter a syntax error or a failing unit test, which you quickly fix. This loop is the fastest, with feedback coming in mere seconds.

The next loop involves running scripts locally or executing more comprehensive unit tests. This process typically takes minutes to complete and provides slightly broader feedback on your code's functionality.

Moving outward, we encounter the loop where you create a pull request after completing a set of commits. This stage involves waiting for code reviews from colleagues or for CI/CD pipelines to run, which can take hours.

The outermost loop represents deploying your code to production. This is where the feedback cycle can extend to days, especially when dealing with scheduled tasks like cron jobs. Imagine deploying a cron job set to run at midnight, only to discover it doesn't execute due to a minor syntax error in your YAML file. You then have to wait another 24 hours to verify your fix, which can be incredibly frustrating.

[Diagram showing a cyclic process of writing code, running code, and CI/CD]

This nested loop structure highlights the varying feedback times at different stages of development. The key to improving engineering productivity lies in compressing these feedback loops, particularly by bringing concerns from the outer loops into the inner ones. One of the most significant issues is that infrastructure-related problems often only surface in the outermost loop, after deployment.

The slow nature of these outer loops often leads developers to write and test code locally, ignoring containerization and cloud environments. While this approach might seem faster initially, it ultimately exacerbates the problem by creating a divergence between local and production environments.

In the following sections, we'll explore how we can leverage technology to bridge this gap and create a development experience that combines the speed of local development with the scalability and robustness of cloud infrastructure.




## Frontend

When it comes to developer productivity, it's good to contrast different types of software engineering. Frontend engineers, in particular, have figured out how to create amazingly fast feedback loops. Their workflow is simple yet effective:

1. Write code in one window
2. Look at the website in another window

This setup allows for immediate gratification. As they write code, it hot reloads, and they can see the changes instantly on the website. There's something very gratifying about doing frontend work. If you haven't tried it, you should – it's actually a lot of fun.

The speed of this feedback loop is what makes frontend development so enjoyable. You get to see the results of your work almost immediately, which is incredibly satisfying. This immediate feedback is something that other areas of software development, particularly in data engineering and science, often lack.

As we think about improving tools for data engineers and scientists, we should consider how we can bring this kind of rapid feedback loop to their workflows. The goal should be to make the experience of writing and deploying data-related code as immediate and gratifying as frontend development.




## Backend

When it comes to backend development, the process often follows a simple yet crucial pattern:

1. Write code
2. Check if it compiles
3. Verify if it passes unit tests
4. Ship it

This workflow is quite different from what frontend developers experience. Frontend engineers have figured out a great setup: they can see immediate results of their code changes on one monitor while writing code on another. The hot reloading feature provides them with an almost instantaneous feedback loop, making frontend development quite gratifying.

Backend developers, on the other hand, often work with a slower feedback cycle. They write code, wait for it to compile (if using a compiled language), run unit tests, and then ship it. This process can be more time-consuming, but it does have its advantages.

Interestingly, many backend developers gravitate towards strongly typed languages or those with robust compilers. While this preference is often attributed to the quality guarantees these languages provide, I believe it's also about the feedback loops. For instance, the Rust compiler gives you immediate feedback on what needs to be fixed, which can be quite satisfying.

Unit tests also play a crucial role in the backend development process. They provide an additional layer of verification before the code is shipped, helping to catch potential issues early in the development cycle.

However, this backend workflow, while functional, still leaves room for improvement when it comes to reducing the time between writing code and seeing it in action. As we continue to evolve our development practices, finding ways to shorten these feedback loops while maintaining code quality will be key to enhancing backend developer productivity and satisfaction.




## Data has super long feedback loops

When working with data, one of the biggest challenges we face is the length of our feedback loops. Unlike frontend developers who can see immediate results from their changes, data engineers and scientists often have to wait much longer to see the effects of their work.

Let's consider the typical workflow for a data professional:

1. Write user job
2. Create PR
3. Deploy
4. Wait for it to run

If the job fails or doesn't produce the expected results, we then have to:

5. Add data or print statements
6. Go back to step 2 and repeat the process

Alternatively, we might:

1. Write code
2. Run it on our local machine
3. Look at results
4. Tweak code
5. Repeat from step 2

This process can be incredibly time-consuming and frustrating. The feedback loops are much longer compared to other types of software development, which can significantly impact productivity and job satisfaction.

For instance, imagine you've built what you believe to be the world's greatest cron job, scheduled to run at midnight. You deploy it and eagerly wait, only to find that it doesn't run at all. Perhaps you forgot a semicolon in the YAML file. Now you have to go back, add that semicolon, redeploy, and wait until the next midnight to see if it works. This kind of delay can be soul-crushing for developers accustomed to quick feedback.

These long feedback loops are often exacerbated by infrastructure concerns that only become apparent in the outermost loops of our development process. Issues that didn't manifest locally suddenly appear once the code is deployed to the cloud. The time it takes to push code to the cloud and test it there can be significant, leading many developers to rely more heavily on local testing. However, this creates a divergence between local and cloud environments, potentially introducing new problems.

To address these challenges, we need to find ways to compress these feedback loops. One approach is to bring concerns from the outermost loops back into the innermost loops of our development process. By doing so, we can create a development experience that feels more like local development while leveraging the scalability and power of the cloud.




## Let's put infrastructure into the feedback loop

When developing data applications, one of the biggest challenges we face is the long feedback loops, especially when it comes to infrastructure. These long loops can significantly slow down development and reduce the joy of writing code. To address this, I propose moving most of the infrastructure work to the cloud, which can bring several benefits:

1. It shifts many tasks from an outer loop to an inner loop, reducing wait times and improving productivity.
2. By ensuring the environment is always the same, we eliminate a whole set of potential issues that can arise from environment differences.
3. We gain access to virtually infinite compute power and storage in the cloud.
4. We no longer have to worry about drivers and GPUs on our local machines.

By moving infrastructure concerns to the cloud, we can compress feedback loops and create a development experience that feels almost like local development, but with the scalability and power of cloud computing. This approach allows us to retain the loop of writing code locally while spinning up containers in the cloud and running our code inside them.

The goal is to make this process so fast that it feels like local development, or even better. Imagine being able to take your local code, spin up a container in the cloud from a pool of workers, and run that code inside an arbitrary container - all in about a second. This level of speed and flexibility can significantly improve the development experience for data engineers and data scientists.

By putting infrastructure into the feedback loop, we can create a more efficient and enjoyable development process, allowing data professionals to focus on solving problems rather than wrestling with infrastructure concerns.




## What are containers?

Containers are a fundamental concept in modern software development and deployment. At their core, containers represent all dependencies as a Linux root filesystem. This means that when you open a container, you'll find a structure similar to a typical Linux system, including directories like `/usr`, `/etc`, and `/lib`.

In addition to the filesystem, containers also include functionality for resource management and, to a limited extent, security. This allows for better isolation and control of applications running within the container.

The concept of containers can be humorously summarized by a popular meme:

"IT WORKS ON MY MACHINE"
"THEN WE'LL SHIP YOUR MACHINE"
"AND THAT IS HOW DOCKER WAS BORN"

While this meme oversimplifies the origin of Docker, it captures the essence of what containers aim to solve: the problem of code working differently in different environments.

Docker is arguably the most well-known implementation of containers, but it's important to note that it's just one implementation of the OCI (Open Container Initiative) container specification. At its core, Docker utilizes a fundamental Linux syscall called `chroot`. This allows Docker to point to a root filesystem inside another root filesystem and run an operating system from that location.

Understanding containers at this level helps us appreciate their power and flexibility in modern software development and deployment workflows.




## Cracking open a Docker container

To understand how Docker containers work, we can examine their contents. A container is essentially a Linux root file system, containing directories like `/usr`, `/etc`, and `/lib`, similar to what you'd find in a Linux VM or physical computer.

Let's walk through the process of inspecting a Docker container:

1. First, we pull a Python container image:
   ```
   $ docker pull python
   ```

2. Then, we run the container in detached mode with a sleep command:
   ```
   $ docker run -d python sleep infinity
   ```

3. Next, we export the container's filesystem to a tar file:
   ```
   $ docker export b0aa33209370 > python.tar
   ```

4. Finally, we can view the contents of the tar file:
   ```
   $ tar tvf python.tar
   ```

This process allows us to see the entire root filesystem of the container. It's worth noting that Docker is just one implementation of the OCI (Open Container Initiative) container spec.

The foundational primitive that Docker uses is the `chroot` syscall in Linux. This allows you to point to a root filesystem inside another root filesystem and run an operating system from there. Docker adds some additional functionality on top of this, such as cgroups for resource isolation.

Understanding the structure of containers is crucial when we start thinking about optimizing container startup times and improving developer productivity. By examining the contents and behavior of containers, we can develop strategies to launch them more efficiently in cloud environments, bringing us closer to the goal of near-instantaneous container deployment for data engineering and machine learning tasks.




## How to launch a container on a remote host

When launching a container on a remote host, there are typically two main steps involved:

1. Pulling down an image: This process can take anywhere from a few seconds to a few minutes, depending on the size of the image and network conditions.
2. Starting the image: This usually takes a couple of seconds.

However, these traditional methods can be too slow for certain use cases, especially when we want to achieve near-local development speeds in a cloud environment. At Modal, we've developed a system that significantly optimizes this process.

Our approach involves several key innovations:

1. Using content addressing to hash and store files efficiently, allowing us to cache common files across different images.
2. Implementing a custom file system using FUSE (Filesystem in Userspace) that fetches files on-demand and caches them locally.
3. Utilizing overlay fs to make the file system writable while maintaining the benefits of our caching system.
4. Pre-computing indexes for container images to speed up the launch process.

By combining these techniques, we've managed to reduce the time it takes to launch a container in the cloud to about a second. This speed improvement allows us to provide a function-as-a-service platform where users can deploy Python functions to the cloud almost instantly, with automatic scaling capabilities.

This approach is particularly beneficial for GPU-intensive workloads, where the ability to quickly scale up and down can lead to significant cost savings. It allows us to offer serverless GPU computing, which is especially valuable for tasks like running inference on large machine learning models with unpredictable load patterns.




## How to launch a container on a remote host

When launching a container on a remote host, there are typically two main steps involved:

1. Pulling down an image: This process can take anywhere from a few seconds to a few minutes, depending on the size of the image and network conditions.

2. Starting the image: This step is generally quicker, usually taking only a couple of seconds.

However, this traditional approach can be too slow for certain use cases, especially when we want to achieve near-local development speeds in a cloud environment. At Modal, we've developed a system that significantly optimizes this process.

Our goal was to take code that exists on a local computer, spin up an arbitrary container in the cloud from a pool of workers, and run that container - all within about a second. To achieve this, we had to rethink the conventional container launching process.

We realized that Docker, while widely used, wasn't the right tool for our needs due to its slow image pulling process. Instead, we turned to a smaller component called `runc`, which Docker uses internally. `runc` doesn't handle image pulling or pushing; it simply starts a container when pointed to a root filesystem.

To make this work, we implemented a custom file system using FUSE (Filesystem in Userspace). This file system uses content addressing, where each file is hashed and stored based on its content. This approach allows us to cache files efficiently, even across different images, significantly reducing the time needed to start a container.

By combining these optimizations with a pool of pre-warmed worker instances in the cloud, we've been able to achieve container start times of about a second, even for custom images. This speed enables us to provide a function-as-a-service platform where users can deploy Python functions to the cloud almost instantly, with automatic scaling capabilities.

This approach has proven particularly valuable for GPU-intensive workloads, where the ability to quickly scale up and down can lead to significant cost savings for users.




## The average container image has a lot of junk

When we look at container images, we often find that they contain a significant amount of unnecessary files. Let's take the Python container from Docker Hub as an example:

- It's 870MB large
- Contains 29,772 files

Breaking this down further, we see:

- 1,553 files in `/usr/share/locale`
- 3,210 files in `/usr/share/doc`
- 1,389 files in `/usr/share/perl`
- 3,050 files in `/usr/share/man`

This abundance of files, many of which are never used, contributes to slower container startup times and increased storage requirements. For instance, there's timezone information for places that your app will likely never need, such as uninhabited islands or Uzbekistan. While each of these files might only be a kilobyte or so, they add up quickly.

Interestingly, when you actually start a container and run a Python script, only a small fraction of these files are accessed. For example, when importing a popular machine learning library like scikit-learn, it reads about 1,000 files and makes approximately 3,000 system calls to stat (a system call to get file information). This means we're only accessing about 3% of the files in the image.

The first files accessed are typically the Python interpreter itself, followed by shared object files (libraries), and then about 800 Python modules. Python also checks for .pyc files (precompiled Python modules), which adds some redundancy to the process.

This realization led us to consider ways to optimize container startup and execution, focusing on accessing only the necessary files and caching them efficiently. By doing so, we can significantly reduce the time it takes to launch and run containers, bringing us closer to a development experience that feels as responsive as local development while leveraging the scalability of the cloud.




## What do we actually need to run something?

When we think about running a Python script or importing a library, we often overlook the underlying file system operations that occur. Let's take a closer look at what happens when we import a popular machine learning library like scikit-learn.

To illustrate this, we can run a simple Python command:

```
$ python3 -c 'import sklearn'
```

This seemingly straightforward import triggers a surprising number of file system operations:

- 3,043 calls to stat
- 1,073 calls to openat

That's a lot of file system operations! However, it's important to note that only a small number of unique files are actually accessed during this process.

This observation has significant implications for how we approach container optimization and fast container startup times. Instead of focusing on reducing the overall size of container images, which often contain many files that are never used, we can concentrate on optimizing access to the specific files that are actually needed when running our code.

Understanding these low-level operations allows us to build more efficient systems for running containers and deploying code in cloud environments. By focusing on the essential files and optimizing their access, we can significantly reduce startup times and improve overall performance.




## What do we actually need to run something?

When we think about running a Python script or importing a library, we often overlook the underlying file system operations that occur. Let's take a closer look at what happens when we import a popular machine learning library like scikit-learn.

Consider this simple Python command:

```python
python3 -c 'import sklearn'
```

Running this command results in:

- 3,043 calls to `stat`
- 1,073 calls to `openat`

That's a lot of file system operations! However, it's important to note that only a small number of unique files are actually accessed.

This observation is crucial when we think about optimizing container startup times and file system access. While there are numerous file operations, they're concentrated on a relatively small subset of files within the container. This insight can lead us to more efficient strategies for container management and file system caching.

In the next section, we'll explore how we can leverage this knowledge to improve container startup times and overall system performance.




## It would be nice to avoid Docker

In our quest to build a better set of tools for data engineers and data scientists, we realized that Docker might not be the ideal solution for our needs. While Docker is widely used for containerization, it comes with some drawbacks, particularly when it comes to speed and efficiency.

Enter runc, a lightweight utility that offers a more streamlined approach to running containers. Runc is a core component of Docker, but it can be used independently. Here's why runc is a nice utility:

1. **Simplicity**: You can point runc at a root file system, and it runs a container. It's that straightforward.
2. **Efficiency**: Runc is not absurdly complex, with only about 50,000 lines of Go code.
3. **Flexibility**: It allows us to bypass some of the overhead associated with Docker while still maintaining containerization benefits.

By using runc directly, we can achieve faster container startup times and more efficient resource utilization. This approach allows us to take code that exists on a local computer, spin up an arbitrary container in the cloud out of a pool of workers, and run that container – all with significantly reduced latency compared to traditional Docker-based solutions.

This shift away from Docker for running containers (while still using it for building images in our initial implementation) marks the beginning of our journey towards creating a more responsive and efficient infrastructure for data teams. It's a step towards compressing feedback loops and bringing infrastructure concerns from the outermost loops back into the innermost loops of development.




## Basic container runner that avoids docker pull

When building a better set of tools for data engineers and data scientists, I realized that one of the key challenges was to optimize the process of running containers in the cloud. The goal was to take code that exists on a local computer, spin up an arbitrary container in the cloud out of a pool of workers, and then run that container—all within about a second.

The traditional approach using Docker proved to be too slow, primarily because pulling down Docker images can take a minute or more, especially for larger images containing libraries like CUDA or TensorFlow. To address this, we developed a basic container runner that avoids the need for `docker pull`.

Our approach consists of two main steps:

1. After building the image:
   - We save the image to a network drive using `docker save`.

2. When running the container:
   - We use `runc` with a root filesystem over the network.

This method allows us to bypass the time-consuming process of pulling down the entire Docker image. Instead, we utilize `runc`, a lightweight component of Docker that can start a container when pointed to a root filesystem and given a JSON configuration.

To implement this, we initially placed the images on NFS (Network File System) and instructed `runc` to start containers from these network-based root filesystems. While this approach was faster than pulling down images, it still suffered from latency issues due to the numerous file operations required when starting a container.

To further optimize performance, we implemented a caching mechanism on the worker executing the containers. By using content addressing and building a custom file system with FUSE, we were able to efficiently cache files locally, dramatically reducing latency and improving container start times.

This basic container runner forms the foundation of a more responsive and efficient system for running arbitrary containers in the cloud, enabling near-instantaneous deployment and execution of code.




## This is still pretty slow though!

While our initial approach of using NFS to store container images improved the speed of container startup, it's still not fast enough to provide a seamless development experience. The main bottleneck is the number of file system operations that Python performs sequentially when importing packages.

Let's take a closer look at the problem:

- Python performs thousands of file system operations sequentially when importing packages.
- NFS latency is typically a few milliseconds per operation.
- This adds up to about 10 seconds of startup time, which is still too slow for our goals.

If we want to achieve container startup times in the range of seconds, we need to reduce the time for each file system operation to a fraction of a millisecond. To put this in perspective, here are some rough latency numbers for different storage options:

- ISI: 30-50ms
- NFS: 1-2ms
- EBS: 0.5-1ms
- SSD: 100-200 μs

To achieve our goal of fast container startup, we need to leverage faster storage options and implement intelligent caching mechanisms. This will allow us to significantly reduce the overall latency and provide a more responsive development experience.




## Can we cache things locally?

When optimizing container startup times, one crucial question arises: can we cache things locally? This approach offers significant potential for improvement, especially when considering the latency of solid-state drives (SSDs).

SSD latency is typically around 100 microseconds (0.1ms), which is much faster than network-based solutions like NFS. This speed advantage becomes particularly important when we consider the file access patterns of container startups.

Interestingly, when launching the same image multiple times, we observe that almost the same files are read every time. This consistency in file access presents an excellent opportunity for caching. Even more surprising is that when launching different images, there's still a significant overlap in the files being accessed.

This overlap occurs because many container images share common components, such as base operating system files, shared libraries, and popular programming language runtimes. By caching these frequently accessed files locally, we can dramatically reduce the time it takes to start a container.

To implement this caching strategy, we developed a custom file system using FUSE (Filesystem in Userspace). This file system keeps an in-memory index of the root filesystem layout, storing both the content hash and metadata for each file. When a file is requested, we first check if it's available in the local SSD cache. If it is, we can serve it immediately, benefiting from both the SSD's low latency and the Linux page cache. If the file isn't cached, we fetch it from a remote service and store it locally for future use.

This approach allows us to achieve high cache efficiency even when users are running arbitrary custom images with no layers in common. By focusing on caching the files that are actually read during container startup, rather than entire images, we can significantly reduce startup times and improve the overall user experience.




## Unrelated images have a lot of overlap!

When working with container images, an interesting observation emerges: even unrelated images have a significant amount of overlap in their file contents. To illustrate this point, I conducted an analysis of three large images from Docker Hub: pytorch/pytorch, tensorflow/tensorflow, and huggingface/transformers-pytorch-gpu.

Despite these images having zero layers in common (which is how Docker typically optimizes caching), a Venn diagram of their file contents reveals substantial overlap. This overlap occurs not just in terms of the number of files, but more importantly, in the files that are actually read when running the containers.

This discovery has important implications for optimizing container startup times and storage efficiency. By leveraging this overlap, we can implement caching strategies that significantly improve performance, even when dealing with seemingly unrelated images.

For instance, if we cache files based on their content rather than their location within the image, we can achieve high cache efficiency across diverse workloads. This approach allows us to store each unique file only once, regardless of how many images it appears in or where it's located within those images.

The benefits of this content-based caching strategy are twofold:

1. Storage optimization: We only need to store each unique file once, even if it's present in thousands of different images.
2. Improved startup times: When launching containers, we can quickly retrieve cached files, even if they're from different images, leading to faster container initialization.

This insight into image overlap has been crucial in developing our approach to fast container startup and efficient resource utilization, allowing us to provide a more responsive and cost-effective platform for running containerized workloads.




## How to cache efficiently: content-addressing

When optimizing container startup times, one key strategy is efficient caching through content-addressing. This approach involves hashing every file by its content and using the hash as the location identifier for that file. This method offers two significant advantages:

1. Storage efficiency: We only need to store each unique file once, even if it appears in multiple images. This drastically reduces storage requirements, especially when dealing with many similar containers.

2. Improved caching: If the same file is accessed across different images, even if it's in different locations, we can cache it effectively. This leads to lower latency when reading files.

To implement this, we built a simple file system using FUSE (Filesystem in Userspace). Contrary to popular belief, building file systems isn't as complex as it might seem. Our initial proof of concept was even written in Python, though we later rewrote it in Rust for better performance.

The file system keeps an in-memory index of the root filesystem layout. For each file location, it stores two pieces of information:
1. The content hash, which defines the location of the file's content
2. A stat struct, containing permission bits, executable status, and other metadata (about 20 bytes in total)

When runc (the container runtime) wants to read a file, our file system looks up the hash in the index. If the file is cached on the local SSD, it's returned immediately. If not, we fetch it from a remote service and store it locally for future use. This approach allows us to benefit from the Linux page cache as well, further improving performance.

To illustrate the efficiency of this method, consider the following example from our slide:

```
Indexes of filesystems                   Storage

/f60ca1 5d62f295
/var/lib/docker/overlay2-l/31.../diff/etc/ca-certificates/update.d/jks-keystore 58584b8
/usr/share/ca-certificates/mozilla/DST_Root_CA_X3.crt 996d6d3
/usr/lib/x86_64-linux-gnu/gio/modules/libdconfsettings.so 68b26b5

/f5f412a3 5a85534
/var/lib/docker/overlay2-l/32.../diff/etc/ca-certificates/update.d/jks-keystore 58584b8
/usr/share/ca-certificates/mozilla/DST_Root_CA_X3.crt 996d6d3
/usr/lib/x86_64-linux-gnu/gio/modules/libdconfsettings.so 68b26b5

/f6595a1 5a85534
/var/lib/docker/overlay2-l/33.../diff/etc/ca-certificates/update.d/jks-keystore 58584b8
/usr/share/ca-certificates/mozilla/DST_Root_CA_X3.crt 996d6d3
/usr/lib/x86_64-linux-gnu/gio/modules/libdconfsettings.so 68b26b5
```

As you can see, multiple filesystem indexes point to the same content hashes in storage. This demonstrates how content-addressing allows us to efficiently store and cache files across different container images, even when they have no layers in common.

By implementing this caching strategy, we've significantly reduced the time it takes to start containers in the cloud, bringing it down to about a second. This improvement in startup time, combined with our custom scheduling and scaling mechanisms, has allowed us to create a function-as-a-service platform that feels almost like local development but with the scalability of the cloud.




## How do we make this work with containers?

To make our system work efficiently with containers, we decided to build our own file system. This approach might sound daunting, but it's actually not as difficult as you might think. In fact, you can even implement a file system using Python with FUSE (Filesystem in Userspace). However, the task becomes significantly easier if you're dealing with a read-only file system.

Our solution involves a combination of components working together:

1. **Worker**: This is where the magic happens. We implement a FUSE-based file system on the worker, which interacts with the container and utilizes the local SSD for caching.

2. **File server**: This component serves files over HTTP, allowing the worker to fetch necessary files that aren't cached locally.

3. **EBS (Elastic Block Store)**: While not directly involved in our file system implementation, EBS provides persistent block-level storage for our EC2 instances.

The key to our approach is content addressing. We hash every file by its content and use the hash as the location identifier for that file. This method offers two significant advantages:

1. Storage efficiency: We only need to store each unique file once, even if it appears in multiple container images.
2. Caching benefits: If the same file is accessed across different images, we can leverage the cache, even if the file appears in different locations within those images.

By implementing this system, we've managed to significantly reduce the time it takes to start containers in the cloud. Instead of waiting for entire Docker images to download, our workers can quickly access the necessary files, either from the local cache or by fetching them on-demand from our file server.

This approach allows us to provide a function-as-a-service platform that feels almost like local development but executes in the cloud with the scalability benefits that come with it. It's particularly beneficial for GPU-intensive workloads, where quick scaling and efficient resource utilization are crucial.




## Fuse operations we need to implement

When building our custom file system using FUSE (Filesystem in Userspace), we only needed to implement a handful of operations to support a read-only file system. This simplified our task considerably compared to implementing a full-featured, writable file system. The key FUSE operations we had to implement were:

1. open
2. read
3. release
4. readdir
5. readdirplus

By focusing on these essential operations, we were able to create a functional file system that met our needs for fast container startup. We use overlay fs on top of our read-only file system to make it writable, which allows us to support running containers without implementing all the complexities of a fully writable file system.

This approach of implementing only the necessary FUSE operations demonstrates that building file systems isn't as daunting as it might seem. Even with just these five operations, we were able to create a system that outperformed traditional methods of container deployment and execution.




## Handle the indirection when reading files

When building our container system, we needed to optimize the process of reading files to achieve fast container startup times. We implemented a content-addressing approach, which has been used since the 1960s and 1970s. This method involves hashing the content of each file and using the hash as the location identifier for that file.

To implement this, we built a simple file system using FUSE (Filesystem in Userspace). While many people think building file systems is difficult, it's actually not that hard. We initially created a proof of concept in Python, but later rewrote it in Rust for better performance.

Our file system keeps an in-memory index of the root filesystem layout. For each file location, the index contains two pieces of information:

1. The hash of the content
2. A struct stat (which includes permission bits and other metadata)

When runc (the container runtime) wants to read a file, our system follows these steps:

1. Look up the file path in the in-memory index
2. Retrieve the corresponding hash and stat information
3. Check if the file is cached on the local SSD
   - If cached, return the file (benefiting from the Linux page cache)
   - If not cached, fetch it from a remote service and store it on the local SSD for future use

This approach allows us to achieve high cache efficiency, even when running arbitrary custom images with no layers in common. By caching files based on their content hash, we can reuse cached files across different images, significantly reducing startup times and improving overall performance.




## When reading a file

When reading a file in our system, we follow a specific process to ensure efficient retrieval and storage of data. This process involves both an index and a storage component, which work together to optimize file access.

1. First, we look up the file's hash in the index. This index contains information about all the files in our system, including their locations and content hashes.

2. Next, we check if the file exists on the local disk:

   a. If the file is not present locally, we fetch it from our remote storage, return its content to the requester, and then store the file on the local disk for future access.
   
   b. If the file already exists on the local disk, we simply return its content without needing to fetch it from remote storage.

This approach allows us to maintain a balance between quick local access and the ability to retrieve files that aren't cached locally. By using content-based addressing (hashing) and local caching, we can significantly reduce latency and improve overall system performance.

The index and storage components work together in this process:

- The index maintains a mapping of file hashes to their locations and metadata.
- The storage component handles both local and remote file storage, ensuring that frequently accessed files are available locally for quick retrieval.

This system enables us to efficiently manage and access files across our distributed infrastructure, providing fast access times for cached content while still allowing for the retrieval of less frequently used files from remote storage.




## Ok but how do we get the images into this?

Now that we've established a way to run containers quickly in the cloud, we need to address how to get the images into this system. Since we already build the containers in the cloud, that's a good starting point. However, we need a more efficient method than traditional Docker image pulling.

Here's the approach we developed, which admittedly started as a rather janky idea:

1. Build images using Docker
2. Use `docker save` to export the image to a temporary directory
3. Compute a checksum for every file in the image
4. Upload any file to NFS that we don't already have stored
5. Build an index that maps file paths to their checksums and `struct stat` information
6. Store this index on NFS as well

This method allows us to efficiently manage and distribute container images across our cloud infrastructure. By using content-addressing (hashing file contents) and storing unique files only once, we can significantly reduce storage requirements and improve caching efficiency.

The index we create contains two crucial pieces of information for each file:
1. The hash, which defines the location of the file's content
2. A `struct stat`, which includes metadata like permission bits and whether the file is executable

When we need to access a file, we look it up in this in-memory index. If the file is already cached on the local SSD, we can return it immediately. If not, we fetch it from the remote service and cache it locally for future use.

While this approach solves many of our problems, it does introduce a new challenge: the process of building this index is quite slow. It can take several minutes to export the container, compute checksums for every file, and store the index. This delay is particularly noticeable when users want to make small changes, like installing a single Python package.

To address this, we'll need to optimize our image building process further, which I'll discuss in the next section.




## Much better idea

When it comes to building container images, I realized there's a much better approach than using Docker. The key insight is that building an image is essentially just running containers. By leveraging this concept, we can significantly improve the process.

Here's how it works:

1. We use OverlayFS to make the image writable. This allows us to modify the filesystem during the build process.
2. We run commands inside a container, just like Docker does with its `RUN` instructions.
3. The changes made during these commands are captured in the OverlayFS layer.
4. After each step, we compute content indexes for the modified files very easily.

This approach has several advantages:

- It's much faster than traditional Docker builds.
- We have more control over the process and can optimize it for our specific needs.
- We can integrate it seamlessly with our custom container runtime.

The only downside is that we need to implement our own Dockerfile parser. However, this isn't as daunting as it might seem. Dockerfiles have a relatively simple structure, and we only need to support a handful of commands like `RUN`, `COPY`, `ENV`, `WORKDIR`, and `ENTRYPOINT`.

Here's a snippet of our Rust code that handles the `RUN` command:

```rust
match command {
    DockerfileCommand::Run(args) => {
        // Execute the command in the container
        // Capture changes in OverlayFS
        // Update content indexes
    },
    // Other commands...
}
```

By implementing our own build process, we've gained the ability to create and modify container images extremely quickly. This speed is crucial for our function-as-a-service platform, allowing us to provide a seamless experience for users who need to build and deploy custom environments rapidly.




## What about scheduling?

Now that we've built the foundation for running and building custom images quickly, as well as maintaining a pool of worker instances, we need to address the question of scheduling. How do we efficiently allocate jobs to our workers?

To recap, here's what we've accomplished so far:

1. We can run custom images very fast, launching containers in the cloud in about a second.
2. We can build custom images very fast, implementing our own Dockerfile parser and image building process.
3. We maintain a pool of worker instances in the cloud, using low-level EC2 APIs for quick auto-scaling.

The next step is to build a scheduling mechanism that takes the work we need to do and assigns it to the workers in the cloud. While this is conceptually similar to what Kubernetes does, we decided to implement our own solution for greater control and efficiency.

Our scheduling system focuses on driving utilization, which is a significant economic incentive in a multi-tenant environment like Modal. We monitor every worker, checking CPU, GPU, and memory usage. Then, we allocate work to workers based on both predicted and current resource utilization.

This approach has evolved into a function-as-a-service platform. Instead of users directly running containers, we provide a higher-level abstraction: functions. In Python, users can write a function and add a decorator, which allows us to deploy it to the cloud in about a second. Users can then call that function without worrying about scaling – it scales up and down automatically, even scaling to zero when not in use.

This function-as-a-service model is particularly useful for GPU workloads. GPUs are expensive, and users often have unpredictable loads. Our serverless GPU offering makes financial sense for many users, as they don't need to provision for peak capacity and let expensive GPU instances sit idle. Instead, we can auto-scale quickly as work comes in, leveraging our fast container boot times and efficient resource allocation.

By combining all these elements – fast container launches, quick image building, efficient worker pool management, and smart scheduling – we've created a system that feels almost better than local development. Users can build and launch containers in the cloud without dealing with the complexities of Docker or Kubernetes, while still benefiting from cloud scalability and the ability to use powerful resources like GPUs on-demand.




## Let's run our own resource pool

At Modal, we decided to run our own resource pool to provide efficient and scalable data infrastructure in the cloud. This approach allows us to have fine-grained control over our resources and optimize for our specific use case. Here's how we manage our resource pool:

1. We launch and terminate instances on both AWS and GCP. This multi-cloud approach gives us flexibility and redundancy.

2. We've optimized our process to launch an instance in about 40 seconds. This quick provisioning time is crucial for maintaining responsiveness in our system.

3. We implement an "overprovisioning" strategy to ensure we always have a bit of spare capacity. This approach helps us handle sudden spikes in demand without introducing delays.

4. One of the key benefits we leverage is multi-tenancy. By running workloads for multiple users on the same infrastructure, we can smooth out overall load and improve resource utilization.

5. To maintain an up-to-date view of our resources, every worker reports its available CPU and memory every 2 seconds. This frequent reporting allows us to make informed decisions about resource allocation and scaling.

By managing our own resource pool, we can provide a more responsive and cost-effective service to our users, especially for workloads with unpredictable demands like GPU-intensive tasks. This approach enables us to offer features like serverless GPUs, which can scale up and down rapidly based on demand, without requiring users to provision and manage their own GPU clusters.




## Turning this into a function-as-a-service platform

After developing the ability to quickly start containers in the cloud, we realized that we could turn this into a function-as-a-service platform. The key idea is to reuse the same container for multiple function calls, which allows us to achieve several important goals:

1. Autoscale on-demand: We can quickly spin up new containers as needed to handle increased workload.
2. Scale down to zero: When there's no demand, we can shut down containers to save resources.
3. Fast cold start: By optimizing container startup times, we can minimize the delay when a new function needs to be executed.

This approach is particularly useful for GPU workloads, where the hardware is expensive and often underutilized. By enabling serverless GPU computing, we can provide a cost-effective solution for users with unpredictable workloads, such as those running inference on GPU.

To implement this function-as-a-service platform, we built a system that includes:

1. An API server that receives requests from clients
2. Multiple workers, each capable of running multiple containers
3. A scheduling mechanism to distribute work across the available workers

The platform allows users to write Python functions and deploy them to the cloud with a simple decorator. These functions can then be called remotely, with the platform handling all the scaling and resource management behind the scenes.

This architecture provides several benefits:

1. Improved developer experience: Users can focus on writing their functions without worrying about infrastructure management.
2. Efficient resource utilization: By pooling resources across multiple users, we can achieve higher overall utilization and lower costs.
3. Rapid scaling: The system can quickly adapt to changing workloads, ensuring that resources are available when needed.

While building this in-house might not be the right choice for most companies, it has allowed us to create a platform that's particularly well-suited for data teams and GPU workloads. By avoiding the complexities of Kubernetes and the limitations of existing serverless platforms, we've been able to offer a solution that combines the ease of use of local development with the scalability of cloud computing.




## What does this let us do?

Now that we've built this infrastructure, what capabilities does it give us? Let me illustrate with an example.

With Modal, we can now build a container image in just a few seconds and store it in our distributed storage. Immediately after that, we can launch 100 containers with this image, all within a span of 5 to 10 seconds. This level of speed and scalability brings us to a point where we're potentially offering an even better experience than local development.

The key advantages are:

1. We can build containers locally without dealing with Docker's complexities.
2. We can launch these containers in the cloud almost instantly.
3. We can scale up and down very quickly.

This combination of features allows us to create a development experience that feels like you're writing code locally, but it actually executes in the cloud with all the scalability benefits that come with it. You can build fan-out operations and enjoy feedback loops that are almost as fast as local development, but with the added power of cloud computing.

To give you a concrete example, many of our users run GPU-intensive tasks like Stable Diffusion and DreamBooth on Modal. But it's not just limited to AI image generation. We see a wide range of use cases, including web scraping, computational biotech, and data pipelines. The platform's flexibility allows it to handle various types of workloads efficiently.

While building this infrastructure in-house might not be the right choice for most companies, for us at Modal, it was necessary to achieve the level of performance and flexibility we desired. We believe that for data teams in particular, solutions like Kubernetes can be cumbersome. Many companies end up building their own data platforms on top of Kubernetes, which often results in leaky abstractions that still require Kubernetes knowledge.

Our approach at Modal was to build from the ground up, avoiding the "wrappers on top of wrappers" problem. This allowed us to create a more streamlined and efficient platform, especially suited for data engineering and machine learning workflows.




## What are some use cases?

At Modal, we've built a powerful infrastructure that enables fast container deployment and execution in the cloud. This technology has opened up a wide range of use cases, particularly in areas that require significant computational resources or scalability.

One of the most popular applications we've seen is in the field of AI image generation. Stable Diffusion and Dreambooth are frequently used on our platform, leveraging the ability to quickly spin up GPU instances as needed. These models, which can be quite large (around 5GB for Stable Diffusion), benefit greatly from our optimized container startup and model loading processes.

Beyond AI image generation, we've observed diverse applications across various domains:

1. Computational biotech: Researchers are using our platform to run complex simulations and data analysis tasks that require substantial computing power.

2. Web scraping: The ability to quickly scale up and down makes our platform ideal for large-scale web scraping operations.

3. Data pipelines: Companies are building and running data processing pipelines that can handle varying workloads efficiently.

These are just a few examples of how our technology is being applied. The key advantage is that users can write Python functions locally, deploy them to the cloud in about a second, and then call these functions without worrying about scaling. The platform handles scaling up and down automatically, even scaling to zero when not in use.

This approach is particularly beneficial for GPU workloads, where the high cost of hardware makes efficient utilization crucial. Our serverless GPU offering allows users to access powerful computing resources on-demand, without the need to provision for peak capacity.

By providing this level of abstraction and efficiency, we're enabling developers and data scientists to focus on their core work, rather than getting bogged down in infrastructure management. It's bringing us closer to the ideal of writing code locally but executing it in the cloud with all the scalability benefits that entails.




## Was it dumb to build this in-house?

When building Modal, we made the decision to create our own container management and execution system rather than relying on existing solutions. This choice might seem questionable at first glance, but there were several compelling reasons for taking this approach.

Firstly, Docker, while widely used, proved to be too slow and limited for our specific needs. We required a system that could start containers in the cloud within seconds, which Docker couldn't achieve due to its image pulling and caching mechanisms.

Secondly, adapting Kubernetes to meet our requirements would have demanded an excessive amount of work. Kubernetes is a powerful tool, but it's often overkill for data teams and can introduce unnecessary complexity. Many companies end up building leaky abstractions on top of Kubernetes, forcing their engineers to learn the intricacies of the system anyway.

Lastly, we considered AWS Lambda, but found it both too expensive and too limited for our use case. Lambda doesn't support GPUs, which was a crucial feature for many of our users running machine learning workloads.

By building our own system, we were able to optimize for our specific needs:

1. Rapid container startup (about 1 second)
2. Efficient caching and file system management
3. Quick scaling, especially for GPU workloads
4. A more intuitive developer experience

While this approach might not be suitable for most companies, it allowed us to create a platform that feels almost better than local development. Users can build and launch containers in the cloud quickly, scale up and down effortlessly, and enjoy the benefits of cloud computing without the typical overhead.

In the end, building this system in-house was the right choice for Modal. It enabled us to offer a unique solution that addresses the specific needs of data engineers and scientists, providing them with a more efficient and enjoyable development experience.

[Image of stacked turtles]




## Error in Anthropic API

Unfortunately, there was an error processing the image for the slide text. The error message states:

```
Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Could not process image'}}
```

This means I don't have the actual slide text to reference. However, based on the transcript, I can provide a summary of the key points discussed in this section of the talk:

I'm the founder of Modal, a company that provides data infrastructure in the cloud. In this talk, I'll be discussing the deep technical rabbit hole I went down while trying to build better tools for data engineers and data scientists. 

My background includes building music recommendation systems at Spotify, creating open-source tools like Luigi (a workflow scheduler) and Annoy (a vector database), and serving as CTO for several years. 

The goal of this talk is to explore why we built certain technologies and how they aim to improve developer productivity, particularly for data teams. I'll be getting quite technical, but I want to start by contextualizing the problem we're trying to solve.

When thinking about developer productivity, it's helpful to consider the nested loops of the development process:

1. The innermost loop: Writing code and fixing syntax errors (seconds)
2. Running scripts locally or unit tests (minutes)
3. Creating pull requests and waiting for CI/CD (hours)
4. Deploying to production and waiting for results (days)

These long feedback loops can ruin the joy of writing code, especially for data teams dealing with cron jobs, model training, and large-scale data processing. Our goal is to compress these feedback loops by bringing infrastructure concerns from the outermost loops into the innermost ones.

In the following sections, I'll dive into the technical details of how we approached this problem at Modal, including our work with containers, file systems, and cloud infrastructure.


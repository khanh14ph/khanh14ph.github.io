---
title: "Understanding CMake: Public, Private, and Interface"
date: 2026-06-05
categories:
  - C++
---

When working with modern CMake (version 3.0+), one of the most important concepts to understand is target-based dependency management. Instead of defining global variables for include directories and linked libraries, modern CMake encourages attaching these properties directly to specific targets (like executables or libraries).

This is primarily done using commands like `target_link_libraries`, `target_include_directories`, and `target_compile_definitions`. 

A critical part of these commands is the visibility specifier: `PUBLIC`, `PRIVATE`, or `INTERFACE`. Understanding the difference between these three is essential for writing clean, modular C++ build systems.

## 1. PRIVATE

When you link a library or add an include directory as `PRIVATE`, you are telling CMake:
> *"My target uses this dependency internally, but it does not expose it in its public headers."*

**Example:**
If `LibraryA` uses `LibraryB` entirely inside its `.cpp` source files, and does not include `LibraryB`'s headers inside `LibraryA`'s own `.h` headers, then `LibraryB` is a `PRIVATE` dependency.

```cmake
add_library(LibraryA src_a.cpp)
target_link_libraries(LibraryA PRIVATE LibraryB)
```

Any executable that links to `LibraryA` will **not** inherit the include directories or link dependencies of `LibraryB`.

## 2. INTERFACE

When you specify a dependency as `INTERFACE`, you are telling CMake:
> *"My target does not need this dependency to compile itself, but anything that links to my target will need it."*

**Example:**
This is extremely common for **header-only libraries**. If `LibraryA` is purely a collection of `.h` files that wrap `LibraryB`, it doesn't compile any object files itself. However, anyone using `LibraryA` must also link against `LibraryB`.

```cmake
add_library(LibraryA INTERFACE)
target_link_libraries(LibraryA INTERFACE LibraryB)
```

## 3. PUBLIC

When you specify a dependency as `PUBLIC`, it is simply a combination of `PRIVATE` and `INTERFACE`. You are telling CMake:
> *"My target uses this dependency internally, AND it exposes it in its public headers, so anyone linking to me needs it too."*

**Example:**
If `LibraryA` includes `LibraryB`'s headers inside its own public `.h` files, then anyone who includes `LibraryA`'s headers will implicitly be including `LibraryB`'s headers. Therefore, `LibraryB` must be `PUBLIC`.

```cmake
add_library(LibraryA src_a.cpp)
target_link_libraries(LibraryA PUBLIC LibraryB)
```

## Summary Table

Here is a quick cheat sheet to remember how properties propagate:

| Specifier | Used to build the target itself? | Propagated to dependents? |
|-----------|----------------------------------|---------------------------|
| **PRIVATE**   | Yes                              | No                        |
| **INTERFACE** | No                               | Yes                       |
| **PUBLIC**    | Yes                              | Yes                       |

By strictly adhering to these visibility specifiers, you prevent "dependency leakage" where unnecessary libraries and include paths pollute your entire project, leading to faster build times and a much cleaner C++ architecture!

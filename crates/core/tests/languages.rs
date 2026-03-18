/// Integration tests for the 9 new Tier 1 language configs.
///
/// Each test verifies that a minimal source snippet for the language produces
/// at least one non-empty chunk via `Chunker::chunk_file`.  This exercises
/// the full path: grammar load → parse → node-kind match → chunk boundary.
use skelesearch_core::chunker::Chunker;

fn assert_language_chunks(filename: &str, source: &str) {
    let chunker = Chunker::default();
    let chunks = chunker.chunk_file(filename, source).unwrap();
    assert!(
        !chunks.is_empty(),
        "expected at least one chunk for {filename}, got none"
    );
}

#[test]
fn java_chunks() {
    assert_language_chunks(
        "Hello.java",
        r#"
import java.util.List;

public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }

    public int add(int a, int b) {
        return a + b;
    }
}
"#,
    );
}

#[test]
fn c_chunks() {
    assert_language_chunks(
        "hello.c",
        r#"
#include <stdio.h>
#include <stdlib.h>

int add(int a, int b) {
    return a + b;
}

int main(void) {
    printf("Hello, %d\n", add(1, 2));
    return 0;
}
"#,
    );
}

#[test]
fn cpp_chunks() {
    assert_language_chunks(
        "hello.cpp",
        r#"
#include <iostream>
#include <string>

class Greeter {
public:
    explicit Greeter(std::string name) : name_(std::move(name)) {}

    void greet() const {
        std::cout << "Hello, " << name_ << std::endl;
    }

private:
    std::string name_;
};

int main() {
    Greeter g("World");
    g.greet();
    return 0;
}
"#,
    );
}

#[test]
fn ruby_chunks() {
    assert_language_chunks(
        "hello.rb",
        r#"
require 'json'
require_relative 'helper'

class Greeter
  def initialize(name)
    @name = name
  end

  def greet
    "Hello, #{@name}!"
  end
end

module Utils
  def self.shout(msg)
    msg.upcase
  end
end
"#,
    );
}

#[test]
fn php_chunks() {
    assert_language_chunks(
        "hello.php",
        r#"<?php
use App\Http\Controllers\Controller;

class HelloController extends Controller
{
    public function index(): string
    {
        return "Hello, World!";
    }

    public function greet(string $name): string
    {
        return "Hello, {$name}!";
    }
}
"#,
    );
}

#[test]
fn csharp_chunks() {
    assert_language_chunks(
        "Hello.cs",
        r#"
using System;
using System.Collections.Generic;

namespace HelloWorld
{
    public class Greeter
    {
        private readonly string _name;

        public Greeter(string name)
        {
            _name = name;
        }

        public string Greet()
        {
            return $"Hello, {_name}!";
        }
    }
}
"#,
    );
}

#[test]
fn kotlin_chunks() {
    assert_language_chunks(
        "Hello.kt",
        r#"
import kotlin.math.max

class Greeter(private val name: String) {
    fun greet(): String = "Hello, $name!"
}

fun add(a: Int, b: Int): Int {
    return a + b
}

object Utils {
    fun shout(msg: String): String = msg.uppercase()
}
"#,
    );
}

#[test]
fn swift_chunks() {
    assert_language_chunks(
        "Hello.swift",
        r#"
import Foundation

class Greeter {
    let name: String

    init(name: String) {
        self.name = name
    }

    func greet() -> String {
        return "Hello, \(name)!"
    }
}

struct Point {
    var x: Double
    var y: Double

    func distance(to other: Point) -> Double {
        let dx = x - other.x
        let dy = y - other.y
        return (dx * dx + dy * dy).squareRoot()
    }
}

protocol Describable {
    func describe() -> String
}
"#,
    );
}

#[test]
fn scala_chunks() {
    assert_language_chunks(
        "Hello.scala",
        r#"
import scala.collection.mutable.ListBuffer

class Greeter(val name: String) {
  def greet(): String = s"Hello, $name!"
}

object Utils {
  def add(a: Int, b: Int): Int = a + b

  def shout(msg: String): String = msg.toUpperCase
}

trait Describable {
  def describe(): String
}
"#,
    );
}

// ---------------------------------------------------------------------------
// Header files: C and C++ headers should also be recognized
// ---------------------------------------------------------------------------

#[test]
fn c_header_chunks() {
    assert_language_chunks(
        "math.h",
        r#"
#ifndef MATH_H
#define MATH_H

#include <stdint.h>

int32_t add(int32_t a, int32_t b);
int32_t multiply(int32_t a, int32_t b);

#endif
"#,
    );
}

#[test]
fn cpp_header_chunks() {
    assert_language_chunks(
        "greeter.hpp",
        r#"
#pragma once
#include <string>

class Greeter {
public:
    explicit Greeter(std::string name);
    std::string greet() const;

private:
    std::string name_;
};
"#,
    );
}

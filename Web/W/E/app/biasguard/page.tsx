// E/app/specrag/page.tsx

"use client";

import React from "react";
import Head from "next/head";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";

export default function SpecRAG() {
  return (
    <>
      <Head>
        <title>Speculative RAG API</title>
        <meta
          name="description"
          content="Enhance retrieval-augmented generation with Speculative RAG, a step-by-step API designed for efficient and accurate outputs."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <main className="bg-white dark:bg-neutral-950 text-neutral-800 dark:text-neutral-100">
        {/* Hero Section */}
        <section className="relative bg-white">
          <div className="max-w-7xl mx-auto px-4 py-20 flex flex-col-reverse md:flex-row items-center">
            <div className="md:w-1/2">
              <h1 className="text-4xl md:text-5xl font-bold mb-6">
                Speculative RAG API
              </h1>
              <p className="text-lg md:text-xl mb-6">
                Optimize your AI outputs with our Speculative RAG API, designed to
                leverage draft-and-verify techniques for faster and more reliable
                responses.
              </p>
              <Link href="/contact" legacyBehavior>
                <motion.a
                  whileHover={{ scale: 1.05 }}
                  transition={{ type: "spring", stiffness: 300, damping: 20 }}
                  className="inline-block bg-blue-600 text-white font-semibold px-6 py-3 rounded-md "
                >
                  Get Started
                </motion.a>
              </Link>
            </div>
            <div className="md:w-5/12 mb-10 md:mb-0">
              <Image
                src="/images/specrag_hero.png"
                alt="Speculative RAG Illustration"
                width={512}
                height={534}
                className="w-7/8 h-auto object-contain"
              />
            </div>
          </div>
        </section>

        {/* Overview Section */}
        <section className="py-16 px-4">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-3xl font-semibold mb-4">Revolutionize Retrieval-Augmented Generation</h2>
            <p className="text-lg mb-8">
              Our Speculative RAG API improves generation accuracy by combining
              specialized drafting with parallel verification, ensuring optimal
              outputs for knowledge-intensive queries.
            </p>
          </div>
        </section>

        {/* Key Features Section */}
        <section className="bg-gray-100 dark:bg-neutral-800 py-16 px-4">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-semibold text-center mb-12">
              Why Choose Speculative RAG?
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {/* Feature 1 */}
              <div className="text-center p-4 bg-white dark:bg-neutral-900 rounded-lg shadow flex flex-col items-center">
                <Image
                  src="/images/parallel_processing.png"
                  alt="Parallel Processing"
                  width={300}
                  height={200}
                  className="mb-4"
                />
                <p className="text-s">
                  Utilize parallel drafting with a specialist LM to generate diverse
                  perspectives from your retrievals, minimizing redundancy.
                </p>
              </div>
              {/* Feature 2 */}
              <div className="text-center p-4 bg-white dark:bg-neutral-900 rounded-lg shadow flex flex-col items-center">
                <Image
                  src="/images/efficiency.png"
                  alt="Efficiency and Speed"
                  width={300}
                  height={200}
                  className="mb-4"
                />
                <p className="text-s">
                  Achieve up to 50% faster response times by delegating drafting to a
                  smaller, optimized model while maintaining accuracy.
                </p>
              </div>
              {/* Feature 3 */}
              <div className="text-center p-4 bg-white dark:bg-neutral-900 rounded-lg shadow flex flex-col items-center">
                <Image
                  src="/images/accuracy.png"
                  alt="Enhanced Accuracy"
                  width={300}
                  height={200}
                  className="mb-4"
                />
                <p className="text-s">
                  Verify drafts with a larger LM to ensure each output is grounded
                  in evidence, reducing errors and hallucinations.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <section className="py-16 px-4">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl font-semibold text-center mb-8">
              How Speculative RAG Works
            </h2>
            <div className="space-y-8">
              {/* Step 1 */}
              <div className="flex flex-col md:flex-row items-center">
                <div className="md:w-1/3 mb-4 md:mb-0">
                  <Image
                    src="/images/step1.png"
                    alt="Step 1: Draft Creation"
                    width={300}
                    height={300}
                    className="mx-auto"
                  />
                </div>
                <div className="md:w-2/3 md:pl-8">
                  <h3 className="text-2xl font-medium mb-2">
                    Step 1: Draft Creation
                  </h3>
                  <p>
                    A smaller, specialized model generates drafts based on clustered
                    subsets of retrieved documents, ensuring diverse insights.
                  </p>
                </div>
              </div>
              {/* Step 2 */}
              <div className="flex flex-col md:flex-row-reverse items-center">
                <div className="md:w-1/3 mb-4 md:mb-0">
                  <Image
                    src="/images/step2.png"
                    alt="Step 2: Parallel Verification"
                    width={300}
                    height={300}
                    className="mx-auto"
                  />
                </div>
                <div className="md:w-2/3 md:pr-8">
                  <h3 className="text-2xl font-medium mb-2">
                    Step 2: Parallel Verification
                  </h3>
                  <p>
                    The drafts are evaluated by a larger model, which scores them
                    based on their alignment with the provided rationale and query.
                  </p>
                </div>
              </div>
              {/* Step 3 */}
              <div className="flex flex-col md:flex-row items-center">
                <div className="md:w-1/3 mb-4 md:mb-0">
                  <Image
                    src="/images/step3.png"
                    alt="Step 3: Best Draft Selection"
                    width={300}
                    height={300}
                    className="mx-auto"
                  />
                </div>
                <div className="md:w-2/3 md:pl-8">
                  <h3 className="text-2xl font-medium mb-2">
                    Step 3: Best Draft Selection
                  </h3>
                  <p>
                    The highest-scoring draft is selected as the final output,
                    combining speed and reliability for your applications.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Call to Action */}
        <section className="bg-blue-600 py-16 px-4 text-white">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-3xl font-semibold mb-4">Ready to Elevate Your AI?</h2>
            <p className="text-lg mb-6">
              Experience unparalleled efficiency and accuracy in retrieval-augmented
              generation. Integrate Speculative RAG today.
            </p>
            <Link href="/contact" legacyBehavior>
              <motion.a
                whileHover={{ scale: 1.05 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                className="inline-block bg-white text-blue-600 font-semibold px-6 py-3 rounded-md shadow hover:bg-gray-100 transition"
              >
                Contact Us
              </motion.a>
            </Link>
          </div>
        </section>
      </main>

      <footer className="bg-gray-200 dark:bg-neutral-900 py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-neutral-600 dark:text-neutral-300">
          &copy; {new Date().getFullYear()} YourCompany. All rights reserved.
        </div>
      </footer>
    </>
  );
}

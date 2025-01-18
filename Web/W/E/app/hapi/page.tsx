// E/app/hapi/page.tsx

"use client";

import React from "react";
import Head from "next/head";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";

export default function HAPI() {
  return (
    <>
      <Head>
        <title>HAPI - Hallucination Prevention API</title>
        <meta
          name="description"
          content="Prevent hallucinations in real time by calling our step-by-step API during your LLM's generation process."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <main className="bg-white dark:bg-neutral-950 text-neutral-800 dark:text-neutral-100">
        {/* Hero Section */}
        <section className="relative bg-white">
          <div className="max-w-7xl mx-auto px-4 py-20 flex flex-col-reverse md:flex-row items-center">
            <div className="md:w-1/2">
              <h1 className="text-4xl md:text-5xl font-bold mb-6">
                HAPI - Hallucination Prevention API
              </h1>
              <p className="text-lg md:text-xl mb-6">
                Ensure your AI models generate more accurate and trustworthy
                outputs by integrating HAPI, our real-time API that intercepts
                each generation step to reduce hallucinations.
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
                src="/images/try4.png"
                alt="HAPI Illustration"
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
            <h2 className="text-3xl font-semibold mb-4">Real-Time Prevention</h2>
            <p className="text-lg mb-8">
              Unlike other approaches that check for inaccuracies <em>after</em> the
              text is fully generated, HAPI hooks into your Large Language
              Model’s inference <strong>step by step</strong>. This proactive
              approach identifies potential hallucinations at their source,
              ensuring more consistent and reliable AI outputs.
            </p>
          </div>
        </section>

        {/* Key Features Section */}
        <section className="bg-gray-100 dark:bg-neutral-800 py-16 px-4">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-semibold text-center mb-12">
              Why Choose HAPI?
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {/* Feature 1 */}
              <div className="text-center p-4 bg-white dark:bg-neutral-900 rounded-lg shadow flex flex-col items-center">
                <Image
                  src="/images/monitor.png"
                  alt="In-Process Monitoring"
                  width={300}
                  height={200}
                  className="mb-4"
                />
                
                <p className="text-s">
                  Keep an eye on every token your LLM generates. Our API flags
                  suspicious patterns before they spiral into lengthy
                  hallucinations.
                </p>
              </div>
              {/* Feature 2 */}
              <div className="text-center p-4 bg-white dark:bg-neutral-900 rounded-lg shadow flex flex-col items-center">
                <Image
                  src="/images/latency.png"
                  alt="Minimal Overhead"
                  width={300}
                  height={200}
                  className="mb-4"
                />
                
                <p className="text-s">
                  HAPI is lightweight, adding only a small fraction of extra
                  compute time per generation step, preserving your model’s
                  overall throughput.
                </p>
              </div>
              {/* Feature 3 */}
              <div className="text-center p-4 bg-white dark:bg-neutral-900 rounded-lg shadow flex flex-col items-center">
                <Image
                  src="/images/card.png"
                  alt="Customizable Rules"
                  width={300}
                  height={200}
                  className="mb-4"
                />
                
                <p className="text-s">
                  Tailor HAPI to your domain or data.
                  Define thresholds and signals that matter most to your use
                  case
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <section className="py-16 px-4">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl font-semibold text-center mb-8">
              How HAPI Works
            </h2>
            <div className="space-y-8">
              {/* Step 1 */}
              <div className="flex flex-col md:flex-row items-center">
                <div className="md:w-1/3 mb-4 md:mb-0">
                  <Image
                    src="/images/9.png"
                    alt="Step 1: Integrate Our API"
                    width={300}
                    height={300}
                    className="mx-auto"
                  />
                </div>
                <div className="md:w-2/3 md:pl-8">
                  <h3 className="text-2xl font-medium mb-2">
                    Step 1: Integrate Our API
                  </h3>
                  <p>
                    Add a simple API call right before each token is generated.
                    This ensures you get a real-time score indicating potential
                    hallucinations or inconsistencies.
                  </p>
                </div>
              </div>
              {/* Step 2 */}
              <div className="flex flex-col md:flex-row-reverse items-center">
                <div className="md:w-1/3 mb-4 md:mb-0">
                  <Image
                    src="/images/10.png"
                    alt="Step 2: Monitor Internal States"
                    width={150}
                    height={150}
                    className="mx-auto"
                  />
                </div>
                <div className="md:w-2/3 md:pr-8">
                  <h3 className="text-2xl font-medium mb-2">
                    Step 2: Monitor Internal States
                  </h3>
                  <p>
                    Our system analyzes internal signals from your LLM’s
                    generation process, identifying early signs of factual
                    drift. This lets you course-correct quickly.
                  </p>
                </div>
              </div>
              {/* Step 3 */}
              <div className="flex flex-col md:flex-row items-center">
                <div className="md:w-1/3 mb-4 md:mb-0">
                  <Image
                    src="/images/hapi-step3.webp"
                    alt="Step 3: Reduce Hallucinations in Real Time"
                    width={150}
                    height={150}
                    className="mx-auto"
                  />
                </div>
                <div className="md:w-2/3 md:pl-8">
                  <h3 className="text-2xl font-medium mb-2">
                    Step 3: Reduce Hallucinations in Real Time
                  </h3>
                  <p>
                    If our score crosses your custom threshold, your application
                    can take immediate action—like nudging the LLM with a clarifying prompt or referencing additional knowledge sources.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Call to Action */}
        <section className="bg-blue-600 py-16 px-4 text-white">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-3xl font-semibold mb-4">Ready to Eliminate Hallucinations?</h2>
            <p className="text-lg mb-6">
              Start your journey towards real-time, step-by-step prevention of
              AI inaccuracies. Sign up and integrate HAPI today.
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

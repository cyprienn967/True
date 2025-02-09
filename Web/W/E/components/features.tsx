"use client";

import React, { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import Image from "next/image";
import Link from "next/link";
import createGlobe from "cobe";
import { cn } from "@/lib/utils";
import Card from "./Card";

function CardContent({ children, className }: { children: React.ReactNode; className?: string; }) {
  return <div className={cn("p-6", className)}>{children}</div>;
}

function CardTitle({ children, className }: { children: React.ReactNode; className?: string; }) {
  return (
    <h3 className={cn("font-sans text-base font-medium tracking-tight text-neutral-700 dark:text-neutral-100", className)}>
      {children}
    </h3>
  );
}

function CardDescription({ children, className }: { children: React.ReactNode; className?: string; }) {
  return (
    <p className={cn("font-sans max-w-xs text-base font-normal tracking-tight mt-2 text-neutral-500 dark:text-neutral-400", className)}>
      {children}
    </p>
  );
}

function CardSkeletonBody({ children, className }: { children: React.ReactNode; className?: string; }) {
  return <div className={cn("overflow-hidden relative w-full h-full", className)}>{children}</div>;
}

function Header({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative w-fit mx-auto p-4 flex items-center justify-center">
      {children}
    </div>
  );
}

export default function Features() {
  return (
    <div id="features" className="w-full mx-auto bg-gray-100 dark:bg-neutral-950 py-20 px-4 md:px-8">
      <Header>
        <h2 className="font-sans text-bold text-xl text-center md:text-4xl w-fit mx-auto font-bold tracking-tight text-neutral-800 dark:text-neutral-100">
          Empower Your AI with Advanced Safeguards and Grounding Techniques
        </h2>
      </Header>

      <p className="max-w-lg text-sm text-neutral-600 text-center mx-auto mt-4 dark:text-neutral-400">
        Ensure your AI models are reliable, unbiased, and well-grounded with our suite of APIs.
      </p>

      <div className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-7xl mx-auto">
        {[
          { title: "JUSTUS - Bias Mitigation API", description: "Detect and mitigate biases in real-time using hidden state analysis to ensure fair and equitable AI generations.", img: "/images/crowd.jpg", link: "/biasguard" },
          { title: "HAPI - Hallucination Prevention API", description: "Prevent hallucinations in AI responses by leveraging hidden states and semantic entropy during generation.", img: "/images/mountain.jpg", link: "/hapi" },
          { title: "ARISTOTLE - Speculative RAG API", description: "Enhance generation accuracy with speculative Retrieval-Augmented Generation techniques for more grounded and reliable outputs.", img: "/images/saltflats.jpeg", link: "/specrag" },
        ].map((feature, index) => (
          <Card key={index} className="flex flex-col justify-between border border-neutral-200 dark:border-neutral-800 rounded-lg hover:scale-105 transform transition-transform duration-300">
            <CardContent className="h-40 flex flex-col justify-between">
              <CardTitle>{feature.title}</CardTitle>
              <CardDescription>{feature.description}</CardDescription>
            </CardContent>
            <CardSkeletonBody className="mt-4">
              <Image src={feature.img} alt={feature.title} width={150} height={150} className="w-80 h-40 object-cover rounded-lg mx-auto" />
            </CardSkeletonBody>
            <div className="text-center mt-4 mb-6">
              <Link href={feature.link} legacyBehavior>
                <a className="inline-block px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition duration-300">Learn More</a>
              </Link>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}

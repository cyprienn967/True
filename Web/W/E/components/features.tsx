// E/components/features.tsx

"use client";

import React, { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import Image from "next/image";
import Link from "next/link";
import createGlobe from "cobe";
import { cn } from "@/lib/utils"; // Adjust if needed
import Card from "./Card";        // Import our separate Card component

// ----------------------------------------------------------------------
// Sub-Components for the Card Layout
// ----------------------------------------------------------------------
function CardContent({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <div className={cn("p-6", className)}>{children}</div>;
}

function CardTitle({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <h3
      className={cn(
        "font-sans text-base font-medium tracking-tight text-neutral-700 dark:text-neutral-100",
        className
      )}
    >
      {children}
    </h3>
  );
}

function CardDescription({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <p
      className={cn(
        "font-sans max-w-xs text-base font-normal tracking-tight mt-2 " +
          "text-neutral-500 dark:text-neutral-400",
        className
      )}
    >
      {children}
    </p>
  );
}

function CardSkeletonBody({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("overflow-hidden relative w-full h-full", className)}>
      {children}
    </div>
  );
}

// Header Component
function Header({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative w-fit mx-auto p-4 flex items-center justify-center">
      <motion.div
        initial={{
          width: 0,
          height: 0,
          borderRadius: 0,
        }}
        whileInView={{
          width: "100%",
          height: "100%",
        }}
        style={{
          transformOrigin: "top-left",
        }}
        transition={{
          duration: 1,
          ease: "easeInOut",
        }}
        className="absolute inset-0 h-full border border-neutral-200 dark:border-neutral-800 w-full rounded-lg"
      >
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.1, ease: "easeInOut" }}
          className="absolute -top-1 -left-1 h-2 w-2 dark:bg-neutral-800 bg-neutral-200 rounded-full"
        />
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.1, ease: "easeInOut" }}
          className="absolute -top-1 -right-1 h-2 w-2 dark:bg-neutral-800 bg-neutral-200 rounded-full"
        />
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.1, ease: "easeInOut" }}
          className="absolute -bottom-1 -left-1 h-2 w-2 dark:bg-neutral-800 bg-neutral-200 rounded-full"
        />
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.1, ease: "easeInOut" }}
          className="absolute -bottom-1 -right-1 h-2 w-2 dark:bg-neutral-800 bg-neutral-200 rounded-full"
        />
      </motion.div>
      {children}
    </div>
  );
}

// (Optional) For fancy animations in skeleton states:
function Container({
  children,
  ...props
}: {
  children: React.ReactNode;
} & React.ComponentProps<typeof motion.div>) {
  return (
    <motion.div
      {...props}
      className={cn(
        "w-full h-14 md:h-40 p-2 rounded-lg relative shadow-lg flex " +
          "items-center bg-black dark:from-neutral-800 " +
          "dark:to-neutral-700 justify-center",
        props.className
      )}
    >
      {children}
    </motion.div>
  );
}

// (Optional) Globe - If you need a rotating globe background
function Globe({ className }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let phi = 0;

    if (!canvasRef.current) return;

    const globe = createGlobe(canvasRef.current, {
      devicePixelRatio: 2,
      width: 600 * 2,
      height: 600 * 2,
      phi: 0,
      theta: 0,
      dark: 1,
      diffuse: 1.2,
      mapSamples: 16000,
      mapBrightness: 6,
      baseColor: [0.0, 0.2, 0.6],
      markerColor: [0, 0, 1],
      glowColor: [1, 1, 1],
      markers: [
        // example markers
        { location: [37.7595, -122.4367], size: 0.03 },
        { location: [40.7128, -74.006], size: 0.1 },
      ],
      onRender: (state) => {
        state.phi = phi;
        phi += 0.01;
      },
    });

    return () => globe.destroy();
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: 600, height: 600, maxWidth: "100%", aspectRatio: 1 }}
      className={className}
    />
  );
}

// ----------------------------------------------------------------------
// Export the 'Features' component as default
// ----------------------------------------------------------------------
export default function Features() {
  return (
    <div
      id="features"
      className="w-full mx-auto bg-gray-100 dark:bg-neutral-950 py-20 px-4 md:px-8"
    >
      <Header>
        <h2 className="font-sans text-bold text-xl text-center md:text-4xl w-fit mx-auto font-bold tracking-tight text-neutral-800 dark:text-neutral-100">
          Empower Your AI with Advanced Safeguards and Grounding Techniques
        </h2>
      </Header>

      <p className="max-w-lg text-sm text-neutral-600 text-center mx-auto mt-4 dark:text-neutral-400">
        Ensure your AI models are reliable, unbiased, and well-grounded with our suite of APIs.
      </p>

      <div className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-7xl mx-auto">
        {/* 1. BiasGuard API */}
        <Link href="/biasguard" legacyBehavior>
        <Card className="flex flex-col justify-between border border-neutral-200 dark:border-neutral-800 rounded-lg hover:scale-105 transform transition-transform duration-300">
            <CardContent className="h-40 flex flex-col justify-between">
              <CardTitle>JUSTUS - Bias Mitigation API</CardTitle>
              <CardDescription>
                Detect and mitigate biases in real-time using hidden state analysis to ensure fair and equitable AI generations.
              </CardDescription>
            </CardContent>
            <CardSkeletonBody className="mt-4">
              <Image
                src="/images/crowd.jpg"
                alt="BiasGuard API"
                width={150}
                height={150}
                className="w-80 h-40 object-cover rounded-lg mx-auto" // Added 'mx-auto' to center
              />
            </CardSkeletonBody>
          </Card>
        </Link>

        {/* 2. HAPI - Hallucination Prevention API */}
        <Link href="/hapi" legacyBehavior>
        <Card className="flex flex-col justify-between border border-neutral-200 dark:border-neutral-800 rounded-lg hover:scale-105 transform transition-transform duration-300">
            <CardContent className="h-40 flex flex-col justify-between">
              <CardTitle>HAPI - Hallucination Prevention API</CardTitle>
              <CardDescription>
                Prevent hallucinations in AI responses by leveraging hidden states and semantic entropy during generation.
              </CardDescription>
            </CardContent>
            <CardSkeletonBody className="mt-4">
              <Image
                src="/images/mountain.jpg"
                alt="HAPI - Hallucination Prevention API"
                width={150}
                height={150}
                className="w-80 h-40 object-cover rounded-lg mx-auto" // Added 'mx-auto' to center
              />
            </CardSkeletonBody>
          </Card>
        </Link>

        {/* 3. SpecRAG API */}
        <Link href="/specrag" legacyBehavior>
        <Card className="flex flex-col justify-between border border-neutral-200 dark:border-neutral-800 rounded-lg hover:scale-105 transform transition-transform duration-300">
            <CardContent className="h-40 flex flex-col justify-between">
              <CardTitle>ARISTOTLE - Speculative RAG API</CardTitle>
              <CardDescription>
                Enhance generation accuracy with speculative Retrieval-Augmented Generation techniques for more grounded and reliable outputs.
              </CardDescription>
            </CardContent>
            <CardSkeletonBody className="mt-4">
              <Image
                src="/images/saltflats.jpeg"
                alt="SpecRAG API"
                width={150}
                height={150}
                className="w-80 h-40 object-cover rounded-lg mx-auto" // Added 'mx-auto' to center
              />
            </CardSkeletonBody>
          </Card>
        </Link>
      </div>
    </div>
  );
}

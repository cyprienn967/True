// E/components/Card.tsx

import React from "react";
import { cn } from "@/lib/utils";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

const Card: React.FC<CardProps> = ({ children, className, ...props }) => {
  return (
    <div className={cn("bg-white dark:bg-neutral-900", className)} {...props}>
      {children}
    </div>
  );
};

export default Card;

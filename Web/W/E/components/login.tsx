"use client";
import React, { useState } from "react";
import { useRouter } from "next/navigation";

export function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const users = {
    admin: { password: "password123" },
    admin2: { password: "password456" },
  };

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();

    const user = users[username];
    if (user && user.password === password) {
      setError(null);
      // Redirect to dashboard with username as query parameter
      router.push(`/dashboard?username=${username}`);
    } else {
      setError("Invalid username or password. Please try again.");
    }
  }

  return (
    <div className="w-full min-h-screen flex items-center justify-center bg-gray-100 dark:bg-neutral-900">
      <form
        className="bg-gray-50 dark:bg-neutral-950 w-full max-w-md p-8 rounded-lg shadow-md"
        onSubmit={onSubmit}
      >
        <div className="text-center">
          <h2 className="text-2xl font-bold leading-9 tracking-tight text-black dark:text-white">
            Log in to your account
          </h2>
        </div>

        <div className="mt-6 space-y-6">
          <div>
            <label
              htmlFor="username"
              className="block text-sm font-medium leading-6 text-neutral-700 dark:text-neutral-400"
            >
              Username
            </label>
            <input
              id="username"
              type="text"
              placeholder="Enter your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="block w-full bg-white dark:bg-neutral-900 px-4 rounded-md border-0 py-1.5 shadow-input text-black placeholder:text-gray-400 focus:ring-2 focus:ring-neutral-400 focus:outline-none sm:text-sm sm:leading-6 dark:text-white"
            />
          </div>

          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium leading-6 text-neutral-700 dark:text-neutral-400"
            >
              Password
            </label>
            <input
              id="password"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="block w-full bg-white dark:bg-neutral-900 px-4 rounded-md border-0 py-1.5 shadow-input text-black placeholder:text-gray-400 focus:ring-2 focus:ring-neutral-400 focus:outline-none sm:text-sm sm:leading-6 dark:text-white"
            />
          </div>

          {error && <div className="text-red-500 text-sm mt-2">{error}</div>}

          <div>
            <button
              type="submit"
              className="bg-black hover:bg-black/90 text-white text-sm transition font-medium duration-200 rounded-full px-4 py-2 flex items-center justify-center w-full dark:text-black dark:bg-white dark:hover:bg-neutral-100"
            >
              Log in
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}

// "use client";
// import React, { useState } from "react";
// import { useRouter } from "next/navigation";

// export function Login() {
//   const [username, setUsername] = useState("");
//   const [password, setPassword] = useState("");
//   const [error, setError] = useState<string | null>(null);
//   const router = useRouter();

//   const users = {
//     admin: { password: "password123" },
//     admin2: { password: "password456" },
//   };

//   async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
//     e.preventDefault();

//     const user = users[username];
//     if (user && user.password === password) {
//       setError(null);
//       // Redirect to dashboard with username as query parameter
//       router.push(`/dashboard?username=${username}`);
//     } else {
//       setError("Invalid username or password. Please try again.");
//     }
//   }

//   return (
//     <form
//       className="bg-gray-50 dark:bg-neutral-950 w-full max-w-md p-8 rounded-lg shadow-md"
//       onSubmit={onSubmit}
//     >
//       <div className="text-center">
//         <h2 className="text-2xl font-bold leading-9 tracking-tight text-black dark:text-white">
//           Log in to your account
//         </h2>
//       </div>

//       <div className="mt-6 space-y-6">
//         <div>
//           <label
//             htmlFor="username"
//             className="block text-sm font-medium leading-6 text-neutral-700 dark:text-neutral-400"
//           >
//             Username
//           </label>
//           <input
//             id="username"
//             type="text"
//             placeholder="Enter your username"
//             value={username}
//             onChange={(e) => setUsername(e.target.value)}
//             className="block w-full bg-white dark:bg-neutral-900 px-4 rounded-md border-0 py-1.5 shadow-input text-black placeholder:text-gray-400 focus:ring-2 focus:ring-neutral-400 focus:outline-none sm:text-sm sm:leading-6 dark:text-white"
//           />
//         </div>

//         <div>
//           <label
//             htmlFor="password"
//             className="block text-sm font-medium leading-6 text-neutral-700 dark:text-neutral-400"
//           >
//             Password
//           </label>
//           <input
//             id="password"
//             type="password"
//             placeholder="••••••••"
//             value={password}
//             onChange={(e) => setPassword(e.target.value)}
//             className="block w-full bg-white dark:bg-neutral-900 px-4 rounded-md border-0 py-1.5 shadow-input text-black placeholder:text-gray-400 focus:ring-2 focus:ring-neutral-400 focus:outline-none sm:text-sm sm:leading-6 dark:text-white"
//           />
//         </div>

//         {error && <div className="text-red-500 text-sm mt-2">{error}</div>}

//         <div>
//           <button
//             type="submit"
//             className="bg-black hover:bg-black/90 text-white text-sm transition font-medium duration-200 rounded-full px-4 py-2 flex items-center justify-center w-full dark:text-black dark:bg-white dark:hover:bg-neutral-100"
//           >
//             Log in
//           </button>
//         </div>
//       </div>
//     </form>
//   );
// }

// SimGameApi.cs -- the sts2 members whose signature differs between the game
// builds the mod supports, reached at runtime so ONE DLL runs on both.
//
// The mod compiles against whichever sts2.dll is installed, but a direct call
// binds the exact IL signature of that build and throws
// MissingMethodException on the other (0.7.0 on the Workshop was built
// against the beta and gave no advice on the stable game). Every member below
// is looked up by name instead; tests/compile/build_mod.sh runs RefCheck
// against every game build present so a new direct reference to a drifted
// member fails the gate.
//
//   member                            v0.107.1                    v0.111.0
//   RunRngSet.Seed                    uint                        ulong
//   Rng read count                    int Counter { get; }        private int _counter
//   new Rng(seed)                     Rng(uint, int = 0)          Rng(ulong = 0)
//   StringHelper.GetDeterministic-    int                         ulong
//     HashCode(string)
//   CardSelectCmd.UseSelector /       (ICardSelector)             (ICardSelector,
//     PushSelector                                                 bool localOnly = false)
//
// Sources: decomp v0.107.1 Core.Random/Rng.cs, Core.Runs/RunRngSet.cs,
// Core.Commands/CardSelectCmd.cs; the same files in v0.111.0.

using System;
using System.Linq;
using System.Reflection;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;

namespace SpireMentor;

internal static class SimGameApi
{
    private const BindingFlags Instance = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
    private const BindingFlags Static = BindingFlags.Static | BindingFlags.Public;

    private static readonly PropertyInfo RunSeedProperty =
        typeof(RunRngSet).GetProperty(nameof(RunRngSet.Seed), Instance)
        ?? throw Missing("RunRngSet.Seed");

    // v0.107.1 exposes the count as a property; v0.111.0 keeps only the field
    // (its public face is ToSerializable().counter, which allocates per call).
    private static readonly PropertyInfo? CounterProperty =
        typeof(Rng).GetProperty("Counter", Instance);
    private static readonly FieldInfo? CounterField =
        CounterProperty == null ? typeof(Rng).GetField("_counter", Instance) : null;

    private static readonly MethodInfo HashMethod =
        typeof(StringHelper).GetMethod(nameof(StringHelper.GetDeterministicHashCode), Static, new[] { typeof(string) })
        ?? throw Missing("StringHelper.GetDeterministicHashCode(string)");

    // The seed constructor: first parameter uint (v0.107.1) or ulong
    // (v0.111.0), every other parameter optional.
    private static readonly ConstructorInfo SeedConstructor =
        typeof(Rng).GetConstructors()
            .FirstOrDefault(c =>
            {
                ParameterInfo[] p = c.GetParameters();
                return p.Length >= 1
                    && (p[0].ParameterType == typeof(uint) || p[0].ParameterType == typeof(ulong))
                    && p.Skip(1).All(x => x.IsOptional);
            })
        ?? throw Missing("Rng(uint|ulong seed)");

    private static readonly MethodInfo UseSelectorMethod = SelectorMethod(nameof(CardSelectCmd.UseSelector));
    private static readonly MethodInfo PushSelectorMethod = SelectorMethod(nameof(CardSelectCmd.PushSelector));

    /// <summary>
    /// The run's once-hashed master seed, boxed as the running build declares
    /// it (<c>uint</c> / <c>ulong</c>); both serialize as a JSON number.
    /// </summary>
    public static object RunSeed(RunRngSet rng) => RunSeedProperty.GetValue(rng)!;

    /// <summary>How many values the stream has produced.</summary>
    public static int Counter(Rng rng) =>
        (int)(CounterProperty != null ? CounterProperty.GetValue(rng)! : CounterField!.GetValue(rng)!);

    /// <summary>
    /// <c>new Rng(seed)</c>, the seed narrowed or widened to the build's type:
    /// v0.107.1 takes the uint truncation of its int hash, v0.111.0 the whole
    /// ulong.
    /// </summary>
    public static Rng NewRng(ulong seed)
    {
        ParameterInfo[] p = SeedConstructor.GetParameters();
        var args = new object?[p.Length];
        args[0] = p[0].ParameterType == typeof(uint) ? (object)unchecked((uint)seed) : seed;
        for (int i = 1; i < p.Length; i++) args[i] = p[i].DefaultValue;
        return (Rng)SeedConstructor.Invoke(args);
    }

    /// <summary>
    /// An Rng seeded the way the game's own AutoSlayer seeds one from a seed
    /// string: v0.107.1 truncates the int hash to uint (AutoSlayer.cs:167),
    /// v0.111.0 passes its ulong hash whole (AutoSlayer.cs:184).
    /// </summary>
    public static Rng RngFromSeedString(string seed)
    {
        object hash = HashMethod.Invoke(null, new object[] { seed })!;
        return NewRng(hash is int i ? unchecked((uint)i) : (ulong)hash);
    }

    /// <summary><c>CardSelectCmd.UseSelector(selector)</c> on the shared stack.</summary>
    public static IDisposable UseSelector(ICardSelector selector) => InvokeSelector(UseSelectorMethod, selector);

    /// <summary><c>CardSelectCmd.PushSelector(selector)</c> on the shared stack.</summary>
    public static IDisposable PushSelector(ICardSelector selector) => InvokeSelector(PushSelectorMethod, selector);

    private static IDisposable InvokeSelector(MethodInfo method, ICardSelector selector)
    {
        ParameterInfo[] p = method.GetParameters();
        var args = new object?[p.Length];
        args[0] = selector;
        // v0.111.0's localOnly defaults to false: the shared stack, which is
        // the only stack v0.107.1 has. In co-op the shared stack answers every
        // player's choice and skips PlayerChoiceSynchronizer; the local one
        // answers only ours and syncs it.
        for (int i = 1; i < p.Length; i++)
        {
            args[i] = p[i].Name == "localOnly" && SimMp.IsMultiplayer ? true : p[i].DefaultValue;
        }
        try
        {
            return (IDisposable)method.Invoke(null, args)!;
        }
        catch (TargetInvocationException ex) when (ex.InnerException != null)
        {
            // Callers catch the game's own exception type ("A card selector
            // is already active."), not the reflection wrapper.
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }
    }

    private static MethodInfo SelectorMethod(string name) =>
        typeof(CardSelectCmd).GetMethods(Static)
            .FirstOrDefault(m =>
            {
                ParameterInfo[] p = m.GetParameters();
                return m.Name == name
                    && p.Length >= 1
                    && p[0].ParameterType == typeof(ICardSelector)
                    && p.Skip(1).All(x => x.IsOptional);
            })
        ?? throw Missing($"CardSelectCmd.{name}(ICardSelector)");

    private static MissingMemberException Missing(string member) =>
        new($"Spire Mentor does not know this game build: {member} has an unexpected shape.");
}
